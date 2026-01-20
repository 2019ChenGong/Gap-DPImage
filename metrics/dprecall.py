from metrics.dp_metrics import DPMetric

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from sklearn.neighbors import NearestNeighbors
from PIL import Image

import os
import shutil
import glob

class DPRecall(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon, noise_multiplier=5.0, clip_bound=10.0, k=6):
        """
        DP-Recall: Differentially private recall metric for generative models.

        Args:
            sensitive_dataset: The private dataset
            public_model: The generative model to evaluate
            epsilon: Privacy budget
            noise_multiplier: Noise multiplier (σ)
            clip_bound: L2 norm clipping bound for features
            k: Number of nearest neighbors to consider (default: 3)
        """
        super().__init__(sensitive_dataset, public_model, epsilon)
        # Load InceptionV3 with transform_input=True
        # This automatically scales images from [0, 1] to [-1, 1] and applies standard normalization
        inception = inception_v3(pretrained=True, transform_input=True).eval()
        inception.fc = nn.Identity()  # Remove the classification head to get 2048-dim features
        self.inception_model = inception.to(self.device)

        # DP parameters
        self.noise_multiplier = noise_multiplier
        self.clip_bound = clip_bound
        self.k = k

        if hasattr(sensitive_dataset, 'dataset'):
            self.dataset_size = len(sensitive_dataset.dataset)
        else:
            self.dataset_size = len(sensitive_dataset) * sensitive_dataset.batch_size

    def _preprocess_images(self, images, is_tensor=True):
        """
        Prepares images for InceptionV3.
        """
        if is_tensor:
            # Handle diffusion model outputs (often in [-1, 1])
            if images.min() < 0:
                images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
        else:
            # Handle NumPy arrays in [0, 255]
            images = torch.from_numpy(images).float() / 255.0
            images = images.permute(0, 3, 1, 2)

        # Standard FID uses Bi-linear interpolation with Anti-aliasing
        images = F.resize(
            images,
            (299, 299),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True
        )

        # Ensure 3 channels (RGB)
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

    def _calculate_recall(self, real_features, generated_features, apply_dp=False):
        """
        Calculate Recall: For each real sample, check if there's a similar generated sample.

        Recall measures coverage: what fraction of real data is "covered" by generated data.
        - Build k-NN index on GENERATED features
        - For each REAL sample, find distance to nearest generated sample
        - Use median of generated-to-generated k-NN distances as threshold
        - Count real samples with nearest-neighbor distance below threshold

        For DP, we add Laplace noise to the count (sensitivity = 1).
        """
        n_real = real_features.shape[0]
        n_gen = generated_features.shape[0]

        print(f"\n   📊 Recall calculation:")
        print(f"      Real samples: {n_real}")
        print(f"      Generated samples: {n_gen}")
        print(f"      k (neighbors): {self.k}")

        # Build k-NN index on GENERATED features
        # Use k+1 for generated-to-generated to exclude self
        nbrs_gen = NearestNeighbors(n_neighbors=min(self.k + 1, n_gen), algorithm='auto', metric='euclidean')
        nbrs_gen.fit(generated_features)

        # Calculate generated-to-generated k-NN distances for threshold
        gen_distances, _ = nbrs_gen.kneighbors(generated_features)
        gen_kth_distances = gen_distances[:, -1]  # k-th neighbor (excluding self)

        # Use median of generated k-NN distances as global threshold
        threshold = np.median(gen_kth_distances)

        # For each real sample, find nearest neighbor in generated set
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
        nbrs.fit(generated_features)
        distances, _ = nbrs.kneighbors(real_features)
        nearest_distances = distances[:, 0]  # Distance to nearest generated sample

        print(f"\n   🎯 Distance statistics:")
        print(f"      Generated k-NN distances - min: {gen_kth_distances.min():.4f}, max: {gen_kth_distances.max():.4f}, median: {np.median(gen_kth_distances):.4f}")
        print(f"      Real-to-Generated NN distances - min: {nearest_distances.min():.4f}, max: {nearest_distances.max():.4f}, mean: {nearest_distances.mean():.4f}")
        print(f"      Threshold: {threshold:.4f}")

        # Count how many real samples have nearest-neighbor distance below threshold
        count_recalled = np.sum(nearest_distances <= threshold)

        if apply_dp:
            # Add DP noise to the count
            # Sensitivity: removing one real sample can change count by at most 1
            sensitivity = 1.0
            noise_scale = sensitivity * self.noise_multiplier

            print(f"\n   🔒 DP protection:")
            print(f"      Count sensitivity: {sensitivity}")
            print(f"      Noise scale: {noise_scale}")

            # Add Laplace noise for count queries
            noise = np.random.laplace(0, noise_scale)
            count_recalled_noisy = count_recalled + noise

            # Clamp to valid range [0, n_real]
            count_recalled_noisy = np.clip(count_recalled_noisy, 0, n_real)

            print(f"      Original count: {count_recalled}")
            print(f"      Noise added: {noise:.4f}")
            print(f"      Noisy count: {count_recalled_noisy:.4f}")

            recall = count_recalled_noisy / n_real
        else:
            recall = count_recalled / n_real

        print(f"\n   📈 Recall calculation details:")
        if apply_dp:
            print(f"      DP-protected count: {count_recalled_noisy:.4f}/{n_real}")
            print(f"      DP-Recall: {recall:.4f}")
        else:
            print(f"      Count: {count_recalled}/{n_real}")
            print(f"      Recall (no DP): {recall:.4f}")

        return recall

    def cal_metric(self, args):
        print("🚀 Starting DP-Recall calculation...")
        print(f"🔒 DP parameters:")
        print(f"   Noise multiplier (σ): {self.noise_multiplier}")
        print(f"   Clipping bound (C): {self.clip_bound}")
        print(f"   Dataset size (n): {self.dataset_size}")
        print(f"   k-nearest neighbors: {self.k}")

        # args.non_DP is False when --non_DP flag is used (store_false action)
        # So apply_dp should be the same as args.non_DP (True by default, False when flag is used)
        apply_dp = args.non_DP

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
            variations_dataloader, is_tensor=True, apply_clipping=apply_dp
        )

        print(f"   Real features shape: {real_features.shape}")
        print(f"   Generated features shape: {generated_features.shape}")

        # Debug: Check feature statistics
        print(f"\n🔍 Feature statistics:")
        print(f"   Real features - mean: {real_features.mean():.4f}, std: {real_features.std():.4f}, min: {real_features.min():.4f}, max: {real_features.max():.4f}")
        print(f"   Generated features - mean: {generated_features.mean():.4f}, std: {generated_features.std():.4f}, min: {generated_features.min():.4f}, max: {generated_features.max():.4f}")
        print(f"   Real feature norms - mean: {np.linalg.norm(real_features, axis=1).mean():.4f}, max: {np.linalg.norm(real_features, axis=1).max():.4f}")
        print(f"   Generated feature norms - mean: {np.linalg.norm(generated_features, axis=1).mean():.4f}, max: {np.linalg.norm(generated_features, axis=1).max():.4f}")

        if apply_dp:
            print("🔐 Applying DP to Recall calculation...")
        recall_score = self._calculate_recall(real_features, generated_features, apply_dp=apply_dp)

        print(f"\n📊 Results:")
        print(f"   Public model: {args.public_model}")
        print(f"   Sensitive dataset: {args.sensitive_dataset}")
        if apply_dp:
            print(f"   DP-Recall Score: {recall_score:.4f}")
            print(f"   Noise multiplier used: {self.noise_multiplier}")
            print(f"   Clipping bound: {self.clip_bound}")
        else:
            print(f"   Recall Score (no DP): {recall_score:.4f}")

        if self.is_delete_variations:
            try:
                if os.path.exists(save_dir):
                    original_dir = os.path.join(save_dir, 'original')
                    variation_dir = os.path.join(save_dir, 'variation')

                    def create_grid(image_dir, output_name):
                        """Create a 10x5 grid from images in directory"""
                        image_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True) +
                                            glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))[:50]
                        if len(image_files) == 0:
                            return None

                        images = [Image.open(f) for f in image_files]
                        n_images = len(images)
                        cols = 10
                        rows = (n_images + cols - 1) // cols
                        img_w, img_h = images[0].size

                        grid = Image.new('RGB', (cols * img_w, rows * img_h))
                        for idx, img in enumerate(images):
                            row = idx // cols
                            col = idx % cols
                            grid.paste(img, (col * img_w, row * img_h))

                        for img in images:
                            img.close()

                        grid_path = os.path.join(save_dir, output_name)
                        grid.save(grid_path)
                        return grid_path, n_images

                    # Create grids for original and variation
                    result_orig = create_grid(original_dir, f"{args.sensitive_dataset}_{args.public_model}_original.png")
                    result_var = create_grid(variation_dir, f"{args.sensitive_dataset}_{args.public_model}_variation.png")

                    if result_orig:
                        print(f"\n📸 Saved {result_orig[1]} original images to: {result_orig[0]}")
                    if result_var:
                        print(f"📸 Saved {result_var[1]} variation images to: {result_var[0]}")

                    # Delete original and variation subdirectories
                    if os.path.exists(original_dir):
                        shutil.rmtree(original_dir)
                    if os.path.exists(variation_dir):
                        shutil.rmtree(variation_dir)
                    print(f"🗑️ Deleted image directories in: {save_dir}")
                else:
                    print(f"\nℹ️ Directory {save_dir} does not exist, no deletion needed.")

            except Exception as e:
                print(f"\n⚠️ Error processing variations: {e}")

        print("\n✅ DP-Recall calculation completed!")
        return recall_score
