import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np


class DPMetric(object):

    def __init__(self, sensitive_dataset, public_model, epsilon):
        self.sensitive_dataset = sensitive_dataset
        self.public_model = public_model 
        self._variation_batch_size = 64
        self._variation_guidance_scale = 7.5
        self._variation_num_inference_steps = 10
        self.epsilon = epsilon
        self.device = "cuda"
        self.max_images = 100
        self.variation_degree = 0.3
    
    def _image_variation(self, images, size=512, variation_degree=None):
        variation_degree=self.variation_degree

        if images.shape[-1] != size:
            original_size = images.shape
            images = F.interpolate(images, size=[size, size])

        max_batch_size = self._variation_batch_size
        variations = []

        num_iterations = int(np.ceil(
            float(images.shape[0]) / max_batch_size))
        prompts = [''] * len(images)
        for iteration in tqdm(range(num_iterations), leave=False):
            variations.append(self.public_model(
                prompt=prompts[iteration * max_batch_size:
                             (iteration + 1) * max_batch_size],
                image=images[iteration * max_batch_size:
                             (iteration + 1) * max_batch_size].to(self.device),
                num_inference_steps=self._variation_num_inference_steps,
                strength=variation_degree,
                guidance_scale=self._variation_guidance_scale,
                num_images_per_prompt=1,
                output_type='np').images)

        # variations = np.concatenate(variations, axis=0)
        # variations = torch.from_numpy(variations).permute(0, 3, 1, 2)
        # variations = F.interpolate(variations, size=[original_size[2], original_size[3]])           

        variations = images, self._round_to_uint8(np.concatenate(variations, axis=0))

        return variations

    def _round_to_uint8(self, image):

        return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)

    def extract_images_from_dataloader(self, dataloader, max_images=None):

        if max_images is None:
            max_images = self.max_images
        
        images = []
        current_count = 0

        
        for batch in dataloader:
            images.append(batch[0])
            current_count += images[-1].shape[0]
            if current_count >= max_images:
                break
        
        # Concatenate all batches into a single tensor
        images = torch.cat(images, dim=0)
        images = images[:max_images]
        
        return images * 2 - 1


    def cal_metric(self):
        print("🚀 Starting DPMetric calculation...")

        extracted_images = self.extract_images_from_dataloader(self.sensitive_dataset, self.max_images)
        print(f"📊 Extracted {len(extracted_images)} images, and extracted image shape: {extracted_images.shape}")
        
        original_images, variations = self._image_variation(extracted_images)
        print(f"📊 Variations shape: {variations.shape}")
        
        print("✅ DPMetric calculation completed!")
        return variations


