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
from metrics.load_public_model import load_public_model
from models.DP_LDM.ldm.models.diffusion.ddim import DDIMSampler
from evaluation.evaluator import get_prompt

def image_variation_batch(rank, dataloader, args):

    size = args.size
    world_size = args.world_size
    variation_save_dir = args.variation_save_dir
    original_save_dir = args.original_save_dir
    max_images = args.max_images
    variation_degree = args.variation_degree
    _variation_num_inference_steps = args._variation_num_inference_steps
    _variation_guidance_scale = args._variation_guidance_scale
    model_id = args.model_id
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    total_processed = 0
    rank_save_dir_variation = os.path.join(variation_save_dir, f"rank_{rank}")
    rank_save_dir_original = os.path.join(original_save_dir, f"rank_{rank}")
    os.makedirs(rank_save_dir_variation, exist_ok=True)
    os.makedirs(rank_save_dir_original, exist_ok=True)
    model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device)
    model.safety_checker = None
    model.requires_safety_checker = False
    model.set_progress_bar_config(disable=True)
    count = 0

    if rank == 0:
        pbar = tqdm(total=max_images, desc="Generating variations", unit="img")

    for images, _ in dataloader:
        batch_size = images.shape[0]
        indices = list(range(batch_size))

        local_indices = indices[rank::world_size]
        if len(local_indices) == 0:
            continue
        local_images = images[local_indices]
        if local_images.shape[1] == 1:
            local_images = local_images.repeat(1, 3, 1, 1)
        original_size = local_images.shape[-2:]  # (H, W)
        prompts = [''] * len(local_images)

        variations = model(
            prompt=prompts,
            image=F.interpolate(local_images, size=[size, size]).to(device) * 2 - 1,
            num_inference_steps=_variation_num_inference_steps,
            strength=variation_degree,
            guidance_scale=_variation_guidance_scale,
            num_images_per_prompt=1,
            output_type='np').images

        variations = torch.from_numpy(variations).permute(0, 3, 1, 2)
        variations = F.interpolate(variations, size=original_size)
        total_processed += batch_size

        for i in range(len(variations)):
            save_image(variations[i], os.path.join(rank_save_dir_variation, f'{count}.png'))
            save_image(local_images[i], os.path.join(rank_save_dir_original, f'{count}.png'))
            count += 1
        
        if rank == 0:
            pbar.update(batch_size)

        if total_processed >= max_images:
            break

def image_generation_batch(rank, args):

    size = args.size
    world_size = args.world_size
    variation_save_dir = args.variation_save_dir
    max_images = args.max_images
    variation_degree = args.variation_degree
    _variation_num_inference_steps = args._variation_num_inference_steps
    _variation_guidance_scale = args._variation_guidance_scale
    model_id = args.model_id
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    total_processed = 0
    rank_save_dir_variation = os.path.join(variation_save_dir, f"rank_{rank}")
    os.makedirs(rank_save_dir_variation, exist_ok=True)
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device)
    model.safety_checker = None
    model.requires_safety_checker = False
    model.set_progress_bar_config(disable=True)
    count = 0

    if rank == 0:
        pbar = tqdm(total=max_images, desc="Generating variations", unit="img")

    batch_size = args._variation_batch_size
    pro = get_prompt(args.data_name)
    while True:
        original_size = (32, 32)
        prompts = pro * (batch_size // world_size // len(pro))

        variations = model(
            prompt=prompts,
            width=512,
            height=512,
            num_inference_steps=_variation_num_inference_steps,
            guidance_scale=_variation_guidance_scale,
            num_images_per_prompt=1,
            output_type='np').images

        variations = torch.from_numpy(variations).permute(0, 3, 1, 2)
        variations = F.interpolate(variations, size=original_size)
        total_processed += batch_size

        for i in range(len(variations)):
            save_image(variations[i], os.path.join(rank_save_dir_variation, f'{count}.png'))
            count += 1
        
        if rank == 0:
            pbar.update(batch_size)

        if total_processed >= max_images:
            break


def image_variation_batch_ldm(rank, dataloader, args):

    size = 32
    world_size = args.world_size
    variation_save_dir = args.variation_save_dir
    original_save_dir = args.original_save_dir
    max_images = args.max_images
    variation_degree = args.variation_degree
    _variation_num_inference_steps = args._variation_num_inference_steps
    _variation_guidance_scale = args._variation_guidance_scale

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    total_processed = 0
    rank_save_dir_variation = os.path.join(variation_save_dir, f"rank_{rank}")
    rank_save_dir_original = os.path.join(original_save_dir, f"rank_{rank}")
    os.makedirs(rank_save_dir_variation, exist_ok=True)
    os.makedirs(rank_save_dir_original, exist_ok=True)
    model = load_public_model('dpimagebench-ldm')
    model = model.to(device)
    sampler = DDIMSampler(model)

    def var_func(images):
        encoder_posterior = model.encode_first_stage(images.to(device))
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        z, _ = sampler.regenerate_image(z, S=_variation_num_inference_steps, start_timestep=int(_variation_num_inference_steps*variation_degree))
        output = model.decode_first_stage(z)
        return torch.clamp(output/2+0.5, min=0.0, max=1.0)

    count = 0

    if rank == 0:
        pbar = tqdm(total=max_images, desc="Generating variations", unit="img")

    for images, _ in dataloader:
        batch_size = images.shape[0]
        indices = list(range(batch_size))

        local_indices = indices[rank::world_size]
        if len(local_indices) == 0:
            continue
        local_images = images[local_indices]
        if local_images.shape[1] == 1:
            local_images = local_images.repeat(1, 3, 1, 1)
        original_size = local_images.shape[-2:]  # (H, W)

        variations = var_func(F.interpolate(local_images, size=[size, size])*2-1)

        variations = F.interpolate(variations, size=original_size)
        total_processed += batch_size

        for i in range(len(variations)):
            save_image(variations[i], os.path.join(rank_save_dir_variation, f'{count}.png'))
            save_image(local_images[i], os.path.join(rank_save_dir_original, f'{count}.png'))
            count += 1
        
        if rank == 0:
            pbar.update(batch_size)

        if total_processed >= max_images:
            break


def image_generation_batch_ldm(rank, args):

    size = 32
    world_size = args.world_size
    variation_save_dir = args.variation_save_dir
    max_images = args.max_images
    variation_degree = args.variation_degree
    _variation_num_inference_steps = 50
    _variation_guidance_scale = args._variation_guidance_scale

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    total_processed = 0
    rank_save_dir_variation = os.path.join(variation_save_dir, f"rank_{rank}")
    os.makedirs(rank_save_dir_variation, exist_ok=True)
    model = load_public_model('dpimagebench-ldm')
    model = model.to(device)
    sampler = DDIMSampler(model)

    def var_func(num, size):
        xc = torch.tensor([0])
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(device)})
        conditioning = c.repeat(num, 1, 1)
        z, _ = sampler.sample(batch_size=num, shape=(3, size, size), S=_variation_num_inference_steps, conditioning=conditioning)
        output = model.decode_first_stage(z)
        return torch.clamp(output/2+0.5, min=0.0, max=1.0)

    count = 0

    if rank == 0:
        pbar = tqdm(total=max_images, desc="Generating variations", unit="img")

    batch_size = args._variation_batch_size
    original_size = (32, 32)
    while True:
        variations = var_func(batch_size//world_size, size)
        variations = F.interpolate(variations, size=original_size)
        total_processed += batch_size

        for i in range(len(variations)):
            save_image(variations[i], os.path.join(rank_save_dir_variation, f'{count}.png'))
            count += 1
        
        if rank == 0:
            pbar.update(batch_size)

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
        self.max_images = 2000
        self.variation_degree = 0.1
        self.is_delete_variations = True
        self.dataloader_size = 16

    def _image_variation(self, dataloader, save_dir, size=512, max_images=100):

        os.makedirs(save_dir, exist_ok=True)
        args = argparse.Namespace()
        args.size = size
        args.variation_save_dir = os.path.join(save_dir, 'variation')
        args.original_save_dir = os.path.join(save_dir, 'original')
        os.makedirs(args.variation_save_dir, exist_ok=True)
        os.makedirs(args.original_save_dir, exist_ok=True)
        args.max_images = max_images
        args.variation_degree = self.variation_degree
        args._variation_num_inference_steps = self._variation_num_inference_steps
        args._variation_guidance_scale = self._variation_guidance_scale
        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise ValueError("Need at least 2 GPUs for multi-GPU generation.")
        args.world_size = world_size
        if hasattr(self.public_model, 'model_id'):
            args.model_id = self.public_model.model_id
            spawn(image_variation_batch, args=(dataloader, args), nprocs=world_size, join=True)
        else:
            args.bench_config = self.public_model.bench_config
            args.ckpt_path = self.public_model.ckpt_path
            spawn(image_variation_batch_ldm, args=(dataloader, args), nprocs=world_size, join=True)

        original_dataset = ImageFolder(args.original_save_dir, transform=transforms.ToTensor())
        variation_dataset = ImageFolder(args.variation_save_dir, transform=transforms.ToTensor())
        return DataLoader(original_dataset, batch_size=self.dataloader_size, shuffle=False), DataLoader(variation_dataset, batch_size=self.dataloader_size, shuffle=False)
    
    def _image_generation(self, save_dir, size=512, max_images=100):

        os.makedirs(save_dir, exist_ok=True)
        args = argparse.Namespace()
        args.size = size
        args.variation_save_dir = os.path.join(save_dir, 'variation')
        os.makedirs(args.variation_save_dir, exist_ok=True)
        args.max_images = max_images
        args.variation_degree = self.variation_degree
        args._variation_num_inference_steps = self._variation_num_inference_steps
        args._variation_guidance_scale = self._variation_guidance_scale
        args._variation_batch_size = self._variation_batch_size
        args.data_name = self.data_name
        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise ValueError("Need at least 2 GPUs for multi-GPU generation.")
        args.world_size = world_size
        if hasattr(self.public_model, 'model_id'):
            args.model_id = self.public_model.model_id
            spawn(image_generation_batch, args=(args,), nprocs=world_size, join=True)
        else:
            args.bench_config = self.public_model.bench_config
            args.ckpt_path = self.public_model.ckpt_path
            spawn(image_generation_batch_ldm, args=(args,), nprocs=world_size, join=True)

        variation_dataset = ImageFolder(args.variation_save_dir, transform=transforms.ToTensor())
        return DataLoader(variation_dataset, batch_size=self.dataloader_size, shuffle=False)

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
        origianl_dataset, variations_dataset = self._image_variation(self.sensitive_dataset, save_dir, max_images=self.max_images)
        print(f"📊 Variations num: {len(variations_dataset.dataset)}")
        
        print("✅ DPMetric calculation completed!")
        return origianl_dataset, variations_dataset


