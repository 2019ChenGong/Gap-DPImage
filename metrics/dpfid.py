from metrics.dp_metrics import DPMetric

import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from scipy import linalg

import os
import shutil
import math

class DPFID(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon, delta=1e-5, clip_bound=10.0):

        super().__init__(sensitive_dataset, public_model, epsilon)
        self.inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)

        # DP parameters
        if hasattr(sensitive_dataset, 'dataset'):
            dataset_size = len(sensitive_dataset.dataset)
        else:
            # Fallback: count batches (less accurate if drop_last=True)
            dataset_size = len(sensitive_dataset) * sensitive_dataset.batch_size
        
        self.delta = 1 / (math.sqrt(dataset_size) * dataset_size)  # delta for (epsilon, delta)-DP
        self.clip_bound = clip_bound  # L2 norm clipping bound for features
        self.privacy_budget_used = 0.0  # Track privacy budget consumption

    def _preprocess_images(self, images, is_tensor=True):
        if is_tensor:
            # images = (images + 1) / 2
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
        features = np.where(clipping_mask, features * (self.clip_bound / norms), features)
        return features

    def _add_gaussian_noise(self, data, sensitivity, epsilon_fraction):
        epsilon_used = self.epsilon * epsilon_fraction
        # Gaussian mechanism: sigma = (sensitivity * sqrt(2 * ln(1.25/delta))) / epsilon
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / epsilon_used
        noise = np.random.normal(0, sigma, size=data.shape)
        self.privacy_budget_used += epsilon_used
        return data + noise

    def _get_inception_features(self, images, is_tensor=True, batch_size=64, apply_clipping=False):
        features = []

        with torch.no_grad():

            for batch, _ in images:
                batch = self._preprocess_images(batch, is_tensor=is_tensor)
                feats = self.inception_model(batch).cpu().numpy()
                features.append(feats)

        features = np.concatenate(features, axis=0)

        # Apply clipping for private dataset features to ensure bounded sensitivity
        if apply_clipping:
            features = self._clip_features(features)

        return features

    def _calculate_fid(self, real_features, generated_features, apply_dp=False):
        n1 = real_features.shape[0]
        n2 = generated_features.shape[0]

        # Compute mean of features
        mu1 = np.mean(real_features, axis=0)
        mu2 = np.mean(generated_features, axis=0)

        if apply_dp:
            # Add DP noise to means
            # Sensitivity of mean with clipped features: clip_bound / n
            mean_sensitivity_1 = self.clip_bound / n1
            mean_sensitivity_2 = self.clip_bound / n2

            # Use 40% of epsilon budget for means (20% each)
            mu1 = self._add_gaussian_noise(mu1, mean_sensitivity_1, epsilon_fraction=0.2)
            mu2 = self._add_gaussian_noise(mu2, mean_sensitivity_2, epsilon_fraction=0.2)
            print(f"   Added DP noise to means (used {0.4 * self.epsilon:.4f} epsilon)")

        # Compute covariance of features
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(generated_features, rowvar=False)

        if apply_dp:
            # Add DP noise to covariances
            # Sensitivity of covariance with clipped features: 2 * clip_bound^2 / n
            # (based on sensitivity analysis for sample covariance)
            cov_sensitivity_1 = 2 * (self.clip_bound ** 2) / n1
            cov_sensitivity_2 = 2 * (self.clip_bound ** 2) / n2

            # Use 60% of epsilon budget for covariances (30% each)
            sigma1 = self._add_gaussian_noise(sigma1, cov_sensitivity_1, epsilon_fraction=0.3)
            sigma2 = self._add_gaussian_noise(sigma2, cov_sensitivity_2, epsilon_fraction=0.3)
            print(f"   Added DP noise to covariances (used {0.6 * self.epsilon:.4f} epsilon)")

            # Ensure covariance matrices are positive semi-definite after noise addition
            sigma1 = self._make_psd(sigma1)
            sigma2 = self._make_psd(sigma2)

        # Compute FID
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def _make_psd(self, matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = 0  # Zero out negative eigenvalues
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def cal_metric(self, args, apply_dp=True):
        print("🚀 Starting DP-FID calculation...")
        print(f"🔒 Privacy parameters: epsilon={self.epsilon}, delta={self.delta}")
        print(f"📏 Feature clipping bound: {self.clip_bound}")

        time = self.get_time()
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Reset privacy budget tracker
        self.privacy_budget_used = 0.0

        # Generate variations
        original_dataloader, variations_dataloader = self._image_variation(
            self.sensitive_dataset, save_dir, max_images=self.max_images
        )
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations: {len(variations_dataloader.dataset)}")

        # Extract Inception V3 features
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

        if apply_dp:
            print("🔐 Applying DP to FID calculation...")
        fid_score = self._calculate_fid(real_features, generated_features, apply_dp=apply_dp)

        print(f"\n📊 Results:")
        print(f"   Public model: {args.public_model}")
        print(f"   Sensitive dataset: {args.sensitive_dataset}")
        if apply_dp:
            print(f"   DP-FID Score: {fid_score:.4f}")
            print(f"   Total privacy budget used: {self.privacy_budget_used:.4f} / {self.epsilon:.4f} epsilon")
            print(f"   Privacy guarantee: ({self.privacy_budget_used:.4f}, {self.delta})-DP")
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