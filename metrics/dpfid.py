from metrics.dp_metrics import DPMetric

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from scipy import linalg
from opacus.accountants.analysis import rdp as privacy_analysis
from PIL import Image

import os
import shutil
import math
import glob

class DPFID(DPMetric):
    def __init__(self, sensitive_dataset, public_model, epsilon, noise_multiplier=5.0, clip_bound=10.0):
        super().__init__(sensitive_dataset, public_model, epsilon)
        
        # Load InceptionV3 with transform_input=True. 
        # This automatically scales images from [0, 1] to [-1, 1] and applies standard normalization.
        inception = inception_v3(pretrained=True, transform_input=True).eval()
        inception.fc = nn.Identity() # Remove the classification head to get 2048-dim features
        self.inception_model = inception.to(self.device)

        self.noise_multiplier = noise_multiplier
        self.clip_bound = clip_bound

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

    def _calculate_fid(self, real_features, generated_features, apply_dp=True):
        """
        Calculates FID between two feature sets.
        If apply_dp is True, adds Gaussian noise to the sufficient statistics.
        """
        n1 = real_features.shape[0]
        n2 = generated_features.shape[0]
        feature_dim = real_features.shape[1]

        if apply_dp:
            # Calculate sum and outer product for DP statistics
            sum_real = np.sum(real_features, axis=0)
            outer_product_real = real_features.T @ real_features
            
            # Usually, generated features are treated as non-private, but we process 
            # them symmetrically here to maintain consistent bias if requested.
            sum_gen = np.sum(generated_features, axis=0)
            outer_product_gen = generated_features.T @ generated_features

            # Add Gaussian noise to sums (Sensitivity = clip_bound)
            sum_noise_std = self.clip_bound * self.noise_multiplier
            sum_real += np.random.normal(0, sum_noise_std, size=sum_real.shape)
            sum_gen += np.random.normal(0, sum_noise_std, size=sum_gen.shape)

            # Add Gaussian noise to outer products (Sensitivity = clip_bound^2)
            # We ensure the noise matrix is symmetric
            op_noise_std = (self.clip_bound ** 2) * self.noise_multiplier
            for op in [outer_product_real, outer_product_gen]:
                noise = np.random.normal(0, op_noise_std, size=(feature_dim, feature_dim))
                op += (noise + noise.T) / np.sqrt(2)

            # Derive mean and covariance from noisy statistics
            mu1, mu2 = sum_real / n1, sum_gen / n2
            sigma1 = (outer_product_real / n1) - np.outer(mu1, mu1)
            sigma2 = (outer_product_gen / n2) - np.outer(mu2, mu2)
            
            # Ensure matrices are Positive Semi-Definite after adding noise
            sigma1 = self._make_psd(sigma1)
            sigma2 = self._make_psd(sigma2)
        else:
            # Standard FID calculation using biased covariance (ddof=0)
            mu1 = np.mean(real_features, axis=0)
            mu2 = np.mean(generated_features, axis=0)
            sigma1 = np.cov(real_features, rowvar=False, ddof=0)
            sigma2 = np.cov(generated_features, rowvar=False, ddof=0)

        # Squared L2 distance between means
        diff = mu1 - mu2
        
        # Matrix square root of the product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors resulting in complex numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Final FID formula: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def _make_psd(self, matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = 0  # Zero out negative eigenvalues
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def cal_metric(self, args):
        print("🚀 Starting DP-FID calculation...")
        print(f"🔒 DP parameters:")
        print(f"   Noise multiplier (σ): {self.noise_multiplier}")
        print(f"   Clipping bound (C): {self.clip_bound}")
        print(f"   Dataset size (n): {self.dataset_size}")

        # Get apply_dp flag (note: non_DP uses store_false, so it's inverted)
        apply_dp = args.non_DP

        time = self.get_time()
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Generate variations
        # original_dataloader, variations_dataloader = self._image_variation(
        #     self.sensitive_dataset, save_dir, max_images=self.max_images
        # )
        original_dataloader, variations_dataloader = self._image_variation(
            self.sensitive_dataset, save_dir, max_images=2000
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

        print("\n✅ DP-FID calculation completed!")
        
        return fid_score