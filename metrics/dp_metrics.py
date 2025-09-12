from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import os
from torch.multiprocessing import spawn
import argparse
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from datetime import datetime

def image_variation_batch(rank, dataloader, args):
    size = args.size
    world_size = args.world_size
    save_dir = args.save_dir
    max_images = args.max_images
    variation_degree = args.variation_degree
    _variation_num_inference_steps = args._variation_num_inference_steps
    _variation_guidance_scale = args._variation_guidance_scale
    model_id = args.model_id
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    total_processed = 0
    rank_save_dir = os.path.join(save_dir, f"rank_{rank}")
    os.makedirs(rank_save_dir, exist_ok=True)
    model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device)
    model.requires_safety_checker = False
    count = 0
    for images, _ in dataloader:
        batch_size = images.shape[0]
        indices = list(range(batch_size))

        local_indices = indices[rank::world_size]
        if len(local_indices) == 0:
            continue
        local_images = images[local_indices].to(device) * 2 - 1
        original_size = local_images.shape[-2:]  # (H, W)
        if local_images.shape[-1] != size:
            local_images = F.interpolate(local_images, size=[size, size])
        prompts = [''] * len(local_images)

        variations = model(
            prompt=prompts,
            image=local_images,
            num_inference_steps=_variation_num_inference_steps,
            strength=variation_degree,
            guidance_scale=_variation_guidance_scale,
            num_images_per_prompt=1,
            output_type='np').images

        variations = torch.from_numpy(variations).permute(0, 3, 1, 2)
        variations = F.interpolate(variations, size=original_size)
        total_processed += batch_size

        for var in variations:
            save_image(var, os.path.join(rank_save_dir, f'{count}.png'))
            count += 1

        if total_processed >= max_images:
            break

class DPMetric(object):

    def __init__(self, sensitive_dataset, public_model, epsilon):
        self.sensitive_dataset = sensitive_dataset
        self.public_model = public_model 
        self._variation_batch_size = 64
        self._variation_guidance_scale = 7.5
        self._variation_num_inference_steps = 10
        self.epsilon = epsilon
        self.device = "cuda"
        self.max_images = 200
        self.variation_degree = 0.1

    def _image_variation(self, dataloader, save_dir, size=512, max_images=100):
        os.makedirs(save_dir, exist_ok=True)
        args = argparse.Namespace()
        args.size = size
        args.save_dir = save_dir
        args.max_images = max_images
        args.variation_degree = self.variation_degree
        args._variation_num_inference_steps = self._variation_num_inference_steps
        args._variation_guidance_scale = self._variation_guidance_scale
        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise ValueError("Need at least 2 GPUs for multi-GPU generation.")
        args.world_size = world_size
        args.model_id = self.public_model.model_id

        spawn(image_variation_batch, args=(dataloader, args), nprocs=world_size, join=True)

        dataset = ImageFolder(save_dir, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=10)

    def _round_to_uint8(self, image):

        return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)

    def get_time(self):

        current_time = datetime.now()
        return current_time.strftime("%Y-%m-%d-%H-%M-%S")


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


    def cal_metric(self, args):
        print("🚀 Starting DPMetric calculation...")

        time = self.get_time()
      
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"
        variations_dataset = self._image_variation(self.sensitive_dataset, save_dir, max_images=self.max_images)
        print(f"📊 Variations num: {len(variations_dataset.dataset)}")
        
        print("✅ DPMetric calculation completed!")
        return variations_dataset


