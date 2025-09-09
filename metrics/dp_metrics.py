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
        self._variation_num_inference_steps = 50
        self.epsilon = epsilon
        self.device = "cuda"
        self.max_images = 500
        self.variation_degree = 0.75
    
    def _image_variation(self, images, size=512, variation_degree=None):
        variation_degree=self.variation_degree
        if images.shape[-1] != size:
            images = F.interpolate(images, target_size=[size, size])

        print(images.shape)
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
        variations = _round_to_uint8(np.concatenate(variations, axis=0))

        return variations

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
        print(f"📊 Extracted {len(extracted_images)} images")
        
        variations = self._image_variation(extracted_images)
        print(f"📊 Variations shape: {variations.shape}")
        
        print("✅ DPMetric calculation completed!")
        return variations

def _round_to_uint8(image):
    return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)

if __name__ == "__main__":

    import os
    os.environ['HF_HOME'] = '/bigtemp/fzv6en/diffuser_cache'
    import requests
    import torch
    from PIL import Image
    from io import BytesIO
    import numpy as np

    from diffusers import StableDiffusionImg2ImgPipeline

    device = "cuda"
    model_id_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")


    init_image = init_image.resize((512, 512))

    init_image = np.array(init_image)
    sensitive_dataset = np.stack([init_image] * 40, axis=0)
    print(sensitive_dataset.shape)

    model = DPMetric(sensitive_dataset=sensitive_dataset, public_model=pipe, epsilon=None)
    _, variations = model.variant()
    print(variations.shape)