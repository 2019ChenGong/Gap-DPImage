import torchvision.transforms as T
from tqdm import tqdm

class DPMetric(object):


    def __init__(self, sensitive_dataset, public_model, epsilon, device):

        self.sensitive_dataset = sensitive_dataset
        self.public_model = public_model #StableDiffusionImg2ImgPipeline
        self._variation_batch_size = 10
        self._variation_guidance_scale = 7.5
        self._variation_num_inference_steps = 50
        self.device = device
        self.epsilon = epsilon

    def variant(self):

        variant_sensitive_dataset = self._image_variation(self.sensitive_dataset)

        return self.sensitive_dataset, variant_sensitive_dataset

    def cal_metric(self):

        pass
    
    def _image_variation(self, images, size=512, variation_degree=0.75):
        width, height = size, size
        variation_transform = T.Compose([
            T.Resize(
                (width, height),
                interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])])
        images = [variation_transform(Image.fromarray(im))
                  for im in images]
        images = torch.stack(images).to(self.device)
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
                             (iteration + 1) * max_batch_size],
                num_inference_steps=self._variation_num_inference_steps,
                strength=variation_degree,
                guidance_scale=self._variation_guidance_scale,
                num_images_per_prompt=1,
                output_type='np').images)
        variations = _round_to_uint8(np.concatenate(variations, axis=0))
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

    model = DPMetric(sensitive_dataset=sensitive_dataset, public_model=pipe, epsilon=None, device=device)
    _, variations = model.variant()
    print(variations.shape)

    # for i, img_array in enumerate(variations):
    #     if img_array.dtype != np.uint8:
    #         img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        
    #     img = Image.fromarray(img_array)
    #     img.save(f"variant_{i:03d}.png")

    # images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    # images[0].save("fantasy_landscape.png")