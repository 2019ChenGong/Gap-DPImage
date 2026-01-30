import os
import json
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from diffusers import StableDiffusionPipeline
from PIL import Image
from typing import List, Union, Tuple
import argparse
import diffusers
import numpy as np
import random
from tqdm import tqdm
from load_attn_procs import load_attn_procs
import shutil

from dataset_bench import get_prompt
import warnings

# 屏蔽特定的 CLIP 截断警告
warnings.filterwarnings("ignore", message="The following part of your input was truncated")

os.environ["attn_update_unet"] = "kqvo"
os.environ["adapter_type"] = "lora"
diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs = load_attn_procs

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def crop_center_width_third(image):
    # 获取图像的尺寸
    width, height = image.size

    # 计算中间1/3的宽度范围
    left = width * 1 // 3   # 左边界（从1/3处开始）
    right = width * 2 // 3  # 右边界（到2/3处结束）

    # 裁剪图像：(left, top, right, bottom)
    cropped_image = image.crop((left, 0, right, height))
    
    return cropped_image

def generate_batch(rank, world_size, args):
    # setup(rank, world_size)
    torch.cuda.set_device(rank)

    device = f"cuda:{rank}"
    dtype = torch.float32

    # === 每个进程创建自己的输出子目录 ===
    image_dir = os.path.join(args.output_dir, "gen")

    # === 分割 prompts 到各 GPU ===
    if type(args.prompts) == str:
        all_prompts = [[args.prompts] for _ in range(args.sample_num)]
    else:
        all_prompts = args.prompts
    all_prompts = all_prompts * args.repeat
    local_prompts = all_prompts[rank::world_size]  # stride 分配
    local_indices = list(range(rank, len(all_prompts), world_size))

    print(f"[GPU {rank}] Handling {len(local_prompts)} prompts")

    # === 加载模型 ===
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None

    # === 加载 LoRA（所有进程都加载）===
    if args.checkpoint_path is not None:
        print(f"Loading LoRA from {args.checkpoint_path}")
        pipe.load_lora_weights(args.checkpoint_path, weight_name="pytorch_lora_weights.safetensors")

    data_entries = []

    with torch.no_grad():
        for i in tqdm(range(0, len(local_prompts), args.batch_size)):
            batch_info = local_prompts[i:i + args.batch_size]
            batch_indices = local_indices[i:i + args.batch_size]

            batch_prompts = [item[0] for item in batch_info]
            batch_labels = [item[1] for item in batch_info]

            # 生成 512x512（SD 原生分辨率）
            gen_h, gen_w = args.gen_size

            images = pipe(
                prompt=batch_prompts,
                height=gen_h,
                width=gen_w,
                num_inference_steps=25,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="pil",
            ).images

            for idx, label, img in zip(batch_indices, batch_labels, images):
                safe_name = f"{idx:04d}"

                # 调整大小到目标分辨率
                if args.crop_middile:
                    img_resized = crop_center_width_third(img.resize(args.target_size, Image.LANCZOS))
                else:
                    img_resized = img.resize(args.target_size, Image.LANCZOS)

                # 保存图像
                lora_path = f"{safe_name}.png"
                img_resized.save(os.path.join(image_dir, f"{label:06d}", lora_path))

    print(f"[GPU {rank}] Completed generation.")

    # cleanup()


def generator_from_prompt_list_multigpu(
    model_id : str,
    output_dir: str,
    prompts: Union[str, List[str]],
    checkpoint_path: str = None,
    sample_num: int = 0,
    repeat: int = 1,
    target_size: Union[int, Tuple[int, int]] = 512,
    gen_size: Union[int, Tuple[int, int]] = 512,
    seed: int = 0,
    guidance_scale: float = 7.0,
    batch_size: int = 4,
    crop_middile: bool = False,
    num_classes: int = 10,
):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise ValueError("Need at least 2 GPUs for multi-GPU generation.")

    print(f"Using {world_size} GPUs for generation.")

    # 构造参数
    args = argparse.Namespace()
    args.model_id = model_id
    args.checkpoint_path = checkpoint_path
    args.output_dir = output_dir
    args.prompts = prompts
    args.target_size = target_size
    args.gen_size = gen_size
    args.batch_size = batch_size
    args.seed = seed
    args.guidance_scale = guidance_scale
    args.sample_num = sample_num
    args.repeat = repeat
    args.crop_middile = crop_middile
    args.num_classes = num_classes

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'gen'), exist_ok=True)
    for cls in range(num_classes):
        os.makedirs(os.path.join(args.output_dir, 'gen', f"{cls:06d}"), exist_ok=True)


    # 启动多进程
    spawn(generate_batch, args=(world_size, args), nprocs=world_size, join=True)

    dataset = ImageFolder(os.path.join(args.output_dir, 'gen'), transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    for x, y in dataloader:
        x = x.numpy()
        y = y.numpy()
        np.savez(os.path.join(args.output_dir, 'gen', 'gen'), x=x, y=y)
        break
    
    show_images = []
    for cls in range(num_classes):
        show_images.append(x[y == cls][:8])
    
    # Concatenate all selected images into a single array
    show_images = np.concatenate(show_images)
    
    # Save the concatenated images as a grid image in the log directory
    torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(args.output_dir, 'gen', 'sample.png'), padding=1, nrow=8)
    
    for cls in range(num_classes):
        shutil.rmtree(os.path.join(args.output_dir, 'gen', f"{cls:06d}"))

    print(f"✅ Multi-GPU generation completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on Flickr30K')
    parser.add_argument('--data_name', type=str, default='CUHK-PEDES')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--target_size', type=int, default=32)
    parser.add_argument('--gen_size', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    args = parser.parse_args()

    prompt_list = get_prompt(args.data_name)
    prompts = [[prompt, idx] for idx, prompt in enumerate(prompt_list)]
    prompts = prompts * (args.num // len(prompt_list))
    # ds = PEDESDataset()
    # print(len(ds))
    # prompts = [example['text'] for example in ds.ds]

    # ds = Flickr30kDataset(image_dir='/bigtemp/fzv6en/kecen/flickr/', json_path='/bigtemp/fzv6en/kecen/dataset_flickr30k.json', split='train')
    # prompts = [example['text'] for example in ds.data]

    # ds = RocoDataset(max_images=None)
    # prompts = [[example['caption']] for example in ds.ds]

    # prompts = [["A man wearing a blue t-shirt, a pair of white and black shorts, a red hat and a bag over his left shoulder"]] * 10
    args.checkpoint_path = args.output_dir
    generator_from_prompt_list_multigpu(
        model_id=args.model_id,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        prompts=prompts,
        batch_size=args.batch_size,
        target_size=(args.target_size, args.target_size),
        gen_size=(args.gen_size, args.gen_size),
        repeat=args.repeat,
        seed=None,
        num_classes=len(prompt_list)
    )