from metrics.dp_metrics import DPMetric

import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from scipy import linalg 

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
        # num_images = images.shape[0]
        # num_batches = int(np.ceil(num_images / batch_size))

        with torch.no_grad():
            # for i in range(num_batches):
            #     batch = images[i * batch_size: (i + 1) * batch_size]
            #     batch = self._preprocess_images(batch, is_tensor=is_tensor)
            #     feats = self.inception_model(batch).cpu().numpy()
            #     features.append(feats)
            
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
        print("🚀 Starting DPMetric calculation...")

        time = self.get_time()
        # save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Extract real images from dataloader
        # extracted_images = self.extract_images_from_dataloader(self.sensitive_dataset, self.max_images)
        # print(f"📊 Extracted {len(extracted_images)} images, and extracted image shape: {extracted_images.shape}")

        # Generate variations
        save_dir = 'exp/test'
        # variations = self._image_variation(self.sensitive_dataset, save_dir)
        original_dataloader, variations_dataloader = self._image_variation(self.sensitive_dataset, save_dir)
        # variations = torch.from_numpy(variations)
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations shape: {len(variations_dataloader.dataset)}")

        # Extract Inception V3 features
        real_features = self._get_inception_features(original_dataloader, is_tensor=True)
        generated_features = self._get_inception_features(variations_dataloader, is_tensor=True)

        # Calculate FID
        fid_score = self._calculate_fid(real_features, generated_features)
        print(f"✅ FID Score: {fid_score}")

        print("✅ DPMetric calculation completed!")
        return fid_score