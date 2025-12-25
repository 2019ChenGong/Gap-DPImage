from metrics.dp_metrics import DPMetric

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from scipy import linalg

import os
import shutil
import math

class InceptionV3FeatureExtractor(nn.Module):
    """Wrapper to extract 2048-dim pool features from Inception V3"""

    def __init__(self, inception_model):
        super().__init__()
        self.inception = inception_model

    def forward(self, x):
        # Forward through Inception V3 up to the avgpool layer
        # See torchvision/models/inception.py for the full forward pass

        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x

class DPFID(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon, noise_multiplier=5.0, clip_bound=20.0):

        super().__init__(sensitive_dataset, public_model, epsilon)
        # Load Inception V3 model and wrap it to extract 2048-dim features
        inception = inception_v3(pretrained=True, transform_input=False).eval()
        self.inception_model = InceptionV3FeatureExtractor(inception).eval().to(self.device)

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

        # Apply ImageNet normalization (required when transform_input=False)
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std

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

    def _add_gaussian_noise(self, data, sensitivity):
        # Using noise multiplier (similar to pe_select.py)
        # Noise std = sensitivity * noise_multiplier
        sigma = sensitivity * self.noise_multiplier
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise

    def _get_inception_features(self, images, is_tensor=True, batch_size=64, apply_clipping=False):
        features = []

        with torch.no_grad():

            for batch, _ in images:
                batch = self._preprocess_images(batch, is_tensor=is_tensor)
                feats = self.inception_model(batch)
                # Features are already (batch, 2048) from InceptionV3FeatureExtractor
                feats = feats.cpu().numpy()
                features.append(feats)

        features = np.concatenate(features, axis=0)

        # Apply clipping for private dataset features to ensure bounded sensitivity
        if apply_clipping:
            features = self._clip_features(features)

        return features

    def _calculate_fid(self, real_features, generated_features, apply_dp=False):
        # n1: actual number of samples used in computation (may be less than dataset_size due to max_images)
        # Sensitivity is computed based on n1, not dataset_size, because it depends on actual samples used
        n1 = real_features.shape[0]
        n2 = generated_features.shape[0]
        feature_dim = real_features.shape[1]

        # Compute mean of features
        mu1 = np.mean(real_features, axis=0)
        mu2 = np.mean(generated_features, axis=0)  # Generated/public data, no noise

        # Compute covariance of features
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(generated_features, rowvar=False)

        if apply_dp:
            # Add DP noise to statistics (mean and covariance)
            # Sensitivity analysis:
            # - Mean sensitivity: ||μ_sensitivity|| = clip_bound / n
            # - Covariance sensitivity: ||Σ_sensitivity|| = 2 × clip_bound² / n

            mean_sensitivity = self.clip_bound / n1
            cov_sensitivity = 2 * (self.clip_bound ** 2) / n1

            print(f"\n   🔒 DP protection via statistic-level noise:")
            print(f"      Mean sensitivity: {mean_sensitivity:.6f}")
            print(f"      Covariance sensitivity: {cov_sensitivity:.6f}")

            # Add Gaussian noise to mean (each dimension independently)
            mean_noise_std = mean_sensitivity * self.noise_multiplier
            mean_noise = np.random.normal(0, mean_noise_std, size=mu1.shape)
            mu1 = mu1 + mean_noise

            print(f"      Mean noise std (per dim): {mean_noise_std:.6f}")
            print(f"      Expected mean noise L2: {np.sqrt(feature_dim) * mean_noise_std:.6f}")

            # Add Gaussian noise to covariance matrix
            # IMPORTANT: Only add noise to diagonal elements to preserve positive definiteness
            # Adding noise to all elements can make the matrix non-PSD, causing trace explosion
            cov_noise_std = cov_sensitivity * self.noise_multiplier

            # Only add noise to diagonal
            diagonal_noise = np.random.normal(0, cov_noise_std, size=feature_dim)
            sigma1 = sigma1 + np.diag(diagonal_noise)

            # Note: diagonal-only noise preserves PSD property, so no need for _make_psd
            # But we keep it for safety in case of numerical errors
            eigvals_before = np.linalg.eigvalsh(sigma1)
            if (eigvals_before < -1e-10).any():
                print(f"      Warning: negative eigenvalues detected, applying _make_psd")
                sigma1 = self._make_psd(sigma1)

            print(f"      Covariance noise std (diagonal only): {cov_noise_std:.6f}")
            print(f"      Expected trace increase from noise: {feature_dim * cov_noise_std:.6f}")

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
            self.sensitive_dataset, save_dir, max_images=self.dataset_size
        )
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations: {len(variations_dataloader.dataset)}")

        print("🔍 Extracting Inception V3 features...")
        # Apply clipping to sensitive/original dataset features for DP
        # Noise will be added at statistic level (mean and covariance)
        real_features = self._get_inception_features(
            original_dataloader, is_tensor=True, apply_clipping=apply_dp
        )
        # Variations are considered public, no clipping or noise needed
        generated_features = self._get_inception_features(
            variations_dataloader, is_tensor=True, apply_clipping=False
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