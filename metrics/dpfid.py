from metrics.dp_metrics import DPMetric

import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from scipy import linalg 

import os
import shutil

class DPFID(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)
        self.inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)

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

    def _get_inception_features(self, images, is_tensor=True, batch_size=64):
        features = []

        with torch.no_grad():
            
            for batch, _ in images:
                batch = self._preprocess_images(batch, is_tensor=is_tensor)
                feats = self.inception_model(batch).cpu().numpy()
                features.append(feats)
        
        return np.concatenate(features, axis=0)

    def _calculate_fid(self, real_features, generated_features):
        # Compute mean of features
        mu1 = np.mean(real_features, axis=0)
        mu2 = np.mean(generated_features, axis=0)
        
        # Compute covariance of features
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(generated_features, rowvar=False)
        
        # Compute FID
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def cal_metric(self, args):
        print("🚀 Starting DPFID calculation...")

        time = self.get_time()
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Generate variations
        # variations = self._image_variation(self.sensitive_dataset, save_dir)
        original_dataloader, variations_dataloader = self._image_variation(self.sensitive_dataset, save_dir, max_images=self.max_images)
        # variations = torch.from_numpy(variations)
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations shape: {len(variations_dataloader.dataset)}")

        # Extract Inception V3 features
        real_features = self._get_inception_features(original_dataloader, is_tensor=True)
        generated_features = self._get_inception_features(variations_dataloader, is_tensor=True)

        # Calculate FID
        fid_score = self._calculate_fid(real_features, generated_features)

        print(f"📊 Public model: {args.public_model}")
        print(f"📊 Sensitive dataset: {args.sensitive_dataset}")
        print(f"✅ FID Score: {fid_score}")
        

        if self.is_delete_variations:
            try:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)  # Recursively delete the directory and its contents
                    print(f"🗑️ Deleted directory: {save_dir}")
                else:
                    print(f"ℹ️ Directory {save_dir} does not exist, no deletion needed.")
            except Exception as e:
                print(f"⚠️ Error deleting directory {save_dir}: {e}")
            print("✅ DPFID calculation completed!")

        else:
            print("✅ DPFID calculation completed!")

        return fid_score