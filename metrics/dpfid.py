from metrics.dp_metrics import DPMetric

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from scipy import linalg
from opacus.accountants.analysis import rdp as privacy_analysis

import os
import shutil
import math

class DPFID(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon, noise_multiplier=5.0, clip_bound=10.0):

        super().__init__(sensitive_dataset, public_model, epsilon)
        # Load Inception V3 and replace fc layer with Identity to get 2048-dim pool features
        inception = inception_v3(pretrained=True, transform_input=False).eval()
        inception.fc = nn.Identity()  # Replace fc layer to output 2048-dim instead of 1000-dim
        self.inception_model = inception.to(self.device)

        # DP parameters - using noise scale instead of privacy budget
        self.noise_multiplier = noise_multiplier  # Noise scale (sigma)
        self.clip_bound = clip_bound  # L2 norm clipping bound for features

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

        features = np.where(clipping_mask, features * (self.clip_bound / norms), features)

        norms_after = np.linalg.norm(features, axis=1, keepdims=True)

        return features

    def _get_inception_features(self, images, is_tensor=True, batch_size=64, apply_clipping=False):
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

    def _calculate_fid(self, real_features, generated_features, apply_dp=True):
        # n1: actual number of samples used in computation (may be less than dataset_size due to max_images)
        # Sensitivity is computed based on n1, not dataset_size, because it depends on actual samples used
        n1 = real_features.shape[0]
        n2 = generated_features.shape[0]
        feature_dim = real_features.shape[1]

        # Compute mean of features
        sum_real = np.sum(real_features, axis=0)
        outer_product_real = real_features.T @ real_features
        sum_gen = np.sum(generated_features, axis=0)
        outer_product_gen = generated_features.T @ generated_features

        if apply_dp:

            # Add Gaussian noise to sum (will divide by n later to get mean)
            # Both real and generated features need noise since variations come from sensitive data
            sum_noise_std = self.clip_bound * self.noise_multiplier
            sum_real += np.random.normal(0, sum_noise_std, size=sum_real.shape)
            sum_gen += np.random.normal(0, sum_noise_std, size=sum_gen.shape)

            # Add Gaussian noise to outer product
            # IMPORTANT: Use independent noise for real and generated features
            outer_product_noise_std = (self.clip_bound ** 2) * self.noise_multiplier

            # Noise for real features
            noise_matrix_real = np.random.normal(0, outer_product_noise_std, size=(feature_dim, feature_dim))
            noise_matrix_real = (noise_matrix_real + noise_matrix_real.T) / np.sqrt(2)
            outer_product_real += noise_matrix_real

            # Independent noise for generated features
            noise_matrix_gen = np.random.normal(0, outer_product_noise_std, size=(feature_dim, feature_dim))
            noise_matrix_gen = (noise_matrix_gen + noise_matrix_gen.T) / np.sqrt(2)
            outer_product_gen += noise_matrix_gen

            mu1 = sum_real / n1
            mu2 = sum_gen / n2

            sigma1 = (outer_product_real / n1) - np.outer(mu1, mu1)
            sigma2 = (outer_product_gen / n2) - np.outer(mu2, mu2)

            print(f"      Sum noise std (per dim): {sum_noise_std:.6f}")
            print(f"      Outer product noise std: {outer_product_noise_std:.6f}")

            # But we keep it for safety in case of numerical errors
            eigvals_before_sigma1 = np.linalg.eigvalsh(sigma1)
            eigvals_before_sigma2 = np.linalg.eigvalsh(sigma2)

            if (eigvals_before_sigma1 < 1e-10).any():
                print(f"      Warning: sigma1 has negative eigenvalues, applying _make_psd")
                sigma1 = self._make_psd(sigma1)
            if (eigvals_before_sigma2 < 1e-10).any():
                print(f"      Warning: sigma2 has negative eigenvalues, applying _make_psd")
                sigma2 = self._make_psd(sigma2)
        else:
            mu1 = np.mean(real_features, axis=0)
            mu2 = np.mean(generated_features, axis=0)

            sigma1 = np.cov(real_features, rowvar=False)
            sigma2 = np.cov(generated_features, rowvar=False)

        # Compute FID using standard formula
        diff = mu1 - mu2
        m = np.square(diff).sum()  # Squared L2 distance between means

        print(f"\n   📊 FID components:")
        print(f"      Mean difference (||μ1-μ2||²): {m:.4f}")
        print(f"      Trace(Σ1): {np.trace(sigma1):.4f}")
        print(f"      Trace(Σ2): {np.trace(sigma2):.4f}")

        # Compute matrix square root of sigma1 @ sigma2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        trace_covmean = np.trace(covmean)
        print(f"      Trace(√(Σ1·Σ2)): {np.real(trace_covmean):.4f}")

        # Calculate FID and ensure it's real-valued
        fid = np.real(m + np.trace(sigma1 + sigma2 - 2 * covmean))

        print(f"      Final FID = {m:.4f} + {np.trace(sigma1):.4f} + {np.trace(sigma2):.4f} - 2×{np.real(trace_covmean):.4f}")

        return fid

    def _make_psd(self, matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = 0  # Zero out negative eigenvalues
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def cal_metric(self, args, apply_dp=True):
        print("🚀 Starting DP-FID calculation...")
        print(f"🔒 DP parameters:")
        print(f"   Noise multiplier (σ): {self.noise_multiplier}")
        print(f"   Clipping bound (C): {self.clip_bound}")
        print(f"   Dataset size (n): {self.dataset_size}")

        time = self.get_time()
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Generate variations
        original_dataloader, variations_dataloader = self._image_variation(
            self.sensitive_dataset, save_dir, max_images=50000
        )
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations: {len(variations_dataloader.dataset)}")

        print("🔍 Extracting Inception V3 features...")
        # Noise will be added at statistic level (mean and covariance)
        real_features = self._get_inception_features(
            original_dataloader, is_tensor=True, apply_clipping=apply_dp
        )
        # Variations are considered public, no clipping or noise needed
        generated_features = self._get_inception_features(
            variations_dataloader, is_tensor=True, apply_clipping=apply_dp
        )

        print(f"   Real features shape: {real_features.shape}")
        print(f"   Generated features shape: {generated_features.shape}")
        print(f"   Actual samples used for DP computation: {real_features.shape[0]}")

        # Debug: Check feature statistics
        print(f"\n🔍 Feature statistics:")
        print(f"   Real features - mean: {real_features.mean():.4f}, std: {real_features.std():.4f}, min: {real_features.min():.4f}, max: {real_features.max():.4f}")
        print(f"   Generated features - mean: {generated_features.mean():.4f}, std: {generated_features.std():.4f}, min: {generated_features.min():.4f}, max: {generated_features.max():.4f}")
        print(f"   Real feature norms - mean: {np.linalg.norm(real_features, axis=1).mean():.4f}, max: {np.linalg.norm(real_features, axis=1).max():.4f}")
        print(f"   Generated feature norms - mean: {np.linalg.norm(generated_features, axis=1).mean():.4f}, max: {np.linalg.norm(generated_features, axis=1).max():.4f}")

        if apply_dp:
            print("🔐 Applying DP to FID calculation...")
        fid_score = self._calculate_fid(real_features, generated_features, apply_dp=apply_dp)

        print(f"\n📊 Results:")
        print(f"   Public model: {args.public_model}")
        print(f"   Sensitive dataset: {args.sensitive_dataset}")
        if apply_dp:
            print(f"   DP-FID Score: {fid_score:.4f}")
            print(f"   Noise multiplier used: {self.noise_multiplier}")
            print(f"   Clipping bound: {self.clip_bound}")
        else:
            print(f"   FID Score (no DP): {fid_score:.4f}")

        if self.is_delete_variations:
            try:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)  # Recursively delete the directory and its contents
                    print(f"\n🗑️ Deleted directory: {save_dir}")
                else:
                    print(f"\nℹ️ Directory {save_dir} does not exist, no deletion needed.")
            except Exception as e:
                print(f"\n⚠️ Error deleting directory {save_dir}: {e}")

        print("\n✅ DP-FID calculation completed!")
        return fid_score