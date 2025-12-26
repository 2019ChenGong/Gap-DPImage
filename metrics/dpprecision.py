from metrics.dp_metrics import DPMetric

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from sklearn.neighbors import NearestNeighbors

import os
import shutil

class DPPrecision(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon, noise_multiplier=5.0, clip_bound=20.0, k=3):
        """
        DP-Precision: Differentially private precision metric for generative models.

        Args:
            sensitive_dataset: The private dataset
            public_model: The generative model to evaluate
            epsilon: Privacy budget
            noise_multiplier: Gaussian noise multiplier (σ)
            clip_bound: L2 norm clipping bound for features
            k: Number of nearest neighbors to consider (default: 3)
        """
        super().__init__(sensitive_dataset, public_model, epsilon)
        # Load Inception V3 and replace fc layer with Identity to get 2048-dim pool features
        inception = inception_v3(pretrained=True, transform_input=False).eval()
        inception.fc = nn.Identity()  # Replace fc layer to output 2048-dim instead of 1000-dim
        self.inception_model = inception.to(self.device)

        # DP parameters
        self.noise_multiplier = noise_multiplier  # Noise scale (σ)
        self.clip_bound = clip_bound  # L2 norm clipping bound for features
        self.k = k  # Number of nearest neighbors

        if hasattr(sensitive_dataset, 'dataset'):
            self.dataset_size = len(sensitive_dataset.dataset)
        else:
            # Fallback: count batches (less accurate if drop_last=True)
            self.dataset_size = len(sensitive_dataset) * sensitive_dataset.batch_size

    def _preprocess_images(self, images, is_tensor=True):
        if is_tensor:
            # Check if images are in [-1, 1] range (common for diffusion models)
            # If so, convert to [0, 1] range for Inception V3
            if images.min() < 0:
                images = (images + 1) / 2
            # Clip to [0, 1] to be safe
            images = torch.clamp(images, 0, 1)
            # Resize to 299x299 for Inception V3
            images = F.resize(images, (299, 299))
        else:
            # Convert NumPy array from [0, 255] to [0, 1]
            images = torch.from_numpy(images).float() / 255.0
            # Resize to 299x299
            images = F.resize(images.permute(0, 3, 1, 2), (299, 299))

        if images.shape[1] != 3:
            images = images.repeat(1, 3, 1, 1)

        return images.to(self.device)

    def _clip_features(self, features):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        clipping_mask = norms > self.clip_bound

        # Debug info
        num_clipped = clipping_mask.sum()
        total_samples = len(features)
        print(f"\n   ✂️  Feature clipping statistics:")
        print(f"      Samples clipped: {num_clipped}/{total_samples} ({100*num_clipped/total_samples:.1f}%)")
        print(f"      Norm before clipping - mean: {norms.mean():.4f}, max: {norms.max():.4f}, min: {norms.min():.4f}")

        features = np.where(clipping_mask, features * (self.clip_bound / norms), features)

        norms_after = np.linalg.norm(features, axis=1, keepdims=True)
        print(f"      Norm after clipping - mean: {norms_after.mean():.4f}, max: {norms_after.max():.4f}")

        return features

    def _get_inception_features(self, images, is_tensor=True, apply_clipping=False):
        features = []

        with torch.no_grad():
            for batch, _ in images:
                batch = self._preprocess_images(batch, is_tensor=is_tensor)
                feats = self.inception_model(batch)
                # Features are (batch, 2048) - fc layer replaced with Identity
                feats = feats.cpu().numpy()
                features.append(feats)

        features = np.concatenate(features, axis=0)

        # Apply clipping for private dataset features to ensure bounded sensitivity
        if apply_clipping:
            features = self._clip_features(features)

        return features

    def _calculate_precision(self, real_features, generated_features, apply_dp=False):
        """
        Calculate Precision: For each generated sample, check if there's a similar real sample.

        Precision = (# of generated samples with at least one close real neighbor) / (# of generated samples)

        IMPORTANT: For DP-Precision, we need to protect the real data.
        The query "find k-NN in real data" reveals information about real data.
        We add noise to the threshold calculation instead of the count.
        """
        n_real = real_features.shape[0]
        n_gen = generated_features.shape[0]

        print(f"\n   📊 Precision calculation:")
        print(f"      Real samples: {n_real}")
        print(f"      Generated samples: {n_gen}")
        print(f"      k (neighbors): {self.k}")

        # Build k-NN index on REAL features (this is sensitive!)
        nbrs = NearestNeighbors(n_neighbors=min(self.k, n_real), algorithm='auto', metric='euclidean')
        nbrs.fit(real_features)

        # For each generated sample, find k nearest neighbors in real set
        distances, indices = nbrs.kneighbors(generated_features)

        # Calculate precision using kth nearest neighbor distance as threshold
        # A generated sample is "precise" if its k-th nearest neighbor in real set is close enough
        kth_distances = distances[:, -1]  # Distance to k-th nearest neighbor

        # Use median distance as threshold (common in Precision/Recall papers)
        threshold = np.median(kth_distances)

        if apply_dp:
            # Add DP noise to threshold calculation
            # Since threshold is derived from real data, it leaks information
            # Sensitivity: changing one real sample can change distances

            # For simplicity, we use a robust DP mechanism:
            # Instead of noisy threshold, we add noise to distance comparisons
            # Sensitivity of median distance: roughly clip_bound / sqrt(n_real)
            threshold_sensitivity = self.clip_bound / np.sqrt(n_real)
            noise_scale = threshold_sensitivity * self.noise_multiplier

            print(f"\n   🔒 DP protection:")
            print(f"      Threshold sensitivity: {threshold_sensitivity:.6f}")
            print(f"      Noise scale: {noise_scale:.6f}")

            # Add Laplace noise to threshold
            noise = np.random.laplace(0, noise_scale)
            threshold_noisy = threshold + noise

            # Ensure threshold is non-negative
            threshold_noisy = max(0, threshold_noisy)

            print(f"      Original threshold: {threshold:.4f}")
            print(f"      Noise added: {noise:.4f}")
            print(f"      Noisy threshold: {threshold_noisy:.4f}")

            threshold = threshold_noisy

        print(f"\n   🎯 Distance statistics:")
        print(f"      Min k-th NN distance: {kth_distances.min():.4f}")
        print(f"      Max k-th NN distance: {kth_distances.max():.4f}")
        print(f"      Mean k-th NN distance: {kth_distances.mean():.4f}")
        if apply_dp:
            print(f"      DP-protected threshold: {threshold:.4f}")
        else:
            print(f"      Median k-th NN distance (threshold): {threshold:.4f}")

        # Count how many generated samples have k-th NN distance below threshold
        count_precise = np.sum(kth_distances <= threshold)

        precision = count_precise / n_gen

        print(f"\n   📈 Precision calculation details:")
        if apply_dp:
            print(f"      Count (using DP threshold): {count_precise}/{n_gen}")
            print(f"      DP-Precision: {precision:.4f}")
        else:
            print(f"      Count: {count_precise}/{n_gen}")
            print(f"      Precision (no DP): {precision:.4f}")

        return precision

    def cal_metric(self, args, apply_dp=True):
        print("🚀 Starting DP-Precision calculation...")
        print(f"🔒 DP parameters:")
        print(f"   Noise multiplier (σ): {self.noise_multiplier}")
        print(f"   Clipping bound (C): {self.clip_bound}")
        print(f"   Dataset size (n): {self.dataset_size}")
        print(f"   k-nearest neighbors: {self.k}")

        time = self.get_time()
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Generate variations
        original_dataloader, variations_dataloader = self._image_variation(
            self.sensitive_dataset, save_dir, max_images=2000
        )
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations: {len(variations_dataloader.dataset)}")

        print("🔍 Extracting Inception V3 features...")
        # Apply clipping to sensitive/original dataset features for DP
        real_features = self._get_inception_features(
            original_dataloader, is_tensor=True, apply_clipping=apply_dp
        )
        # Variations are considered public, no clipping needed
        generated_features = self._get_inception_features(
            variations_dataloader, is_tensor=True, apply_clipping=False
        )

        print(f"   Real features shape: {real_features.shape}")
        print(f"   Generated features shape: {generated_features.shape}")

        # Debug: Check feature statistics
        print(f"\n🔍 Feature statistics:")
        print(f"   Real features - mean: {real_features.mean():.4f}, std: {real_features.std():.4f}")
        print(f"   Generated features - mean: {generated_features.mean():.4f}, std: {generated_features.std():.4f}")
        print(f"   Real feature norms - mean: {np.linalg.norm(real_features, axis=1).mean():.4f}")
        print(f"   Generated feature norms - mean: {np.linalg.norm(generated_features, axis=1).mean():.4f}")

        if apply_dp:
            print("🔐 Applying DP to Precision calculation...")
        precision_score = self._calculate_precision(real_features, generated_features, apply_dp=apply_dp)

        print(f"\n📊 Results:")
        print(f"   Public model: {args.public_model}")
        print(f"   Sensitive dataset: {args.sensitive_dataset}")
        if apply_dp:
            print(f"   DP-Precision Score: {precision_score:.4f}")
            print(f"   Noise multiplier used: {self.noise_multiplier}")
            print(f"   Clipping bound: {self.clip_bound}")
        else:
            print(f"   Precision Score (no DP): {precision_score:.4f}")

        if self.is_delete_variations:
            try:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                    print(f"\n🗑️ Deleted directory: {save_dir}")
                else:
                    print(f"\nℹ️ Directory {save_dir} does not exist, no deletion needed.")
            except Exception as e:
                print(f"\n⚠️ Error deleting directory {save_dir}: {e}")

        print("\n✅ DP-Precision calculation completed!")
        return precision_score
