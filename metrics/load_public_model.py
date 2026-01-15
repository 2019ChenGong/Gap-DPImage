import torch
import random
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset
from data.stylegan3.dataset import ImageFolderDataset
from models.DP_LDM.ldm.util import instantiate_from_config
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler
from omegaconf import OmegaConf

import os

os.environ['HF_HOME'] = '/bigtemp/fzv6en/diffuser_cache'

def load_ldm_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, weights_only=False)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_public_model(public_model):

    if public_model == 'stable-diffusion-v1-5':
        print(f"Loading model: {public_model}")
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'stable-diffusion-2-1-base':
        print(f"Loading model: {public_model}")
        model_id = "Manojb/stable-diffusion-2-1-base"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'stable-diffusion-v1-4':
        print(f"Loading model: {public_model}")
        model_id = "CompVis/stable-diffusion-v1-4"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'stable-diffusion-2-base':
        print(f"Loading model: {public_model}")
        model_id = "Manojb/stable-diffusion-2-base"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'realistic-vision-v5.1':
        print(f"Loading model: {public_model}")
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'realistic-vision-v6.0':
        print(f"Loading model: {public_model}")
        model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'prompt2med':
        print(f"Loading model: {public_model}")
        model_id = "Nihirc/Prompt2MedImage"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'stable-diffusion-2':
        print(f"Loading model: {public_model}")
        model_id = "stabilityai/stable-diffusion-2"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.safety_checker = None
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")
    elif public_model == 'dpimagebench-ldm':
        print(f"Loading model: {public_model}")
        config_path = '/p/fzv6enresearch/gap/models/DP_LDM/configs/finetuning/32_4M.yaml'
        ckpt = '/p/fzv6enresearch/gap/exp/2024-12-11T11-06-26_imagenet32-conditional_ours/checkpoints/last.ckpt'
        configs = [OmegaConf.load(config_path)]
        config = OmegaConf.merge(*configs)
        model = load_ldm_model_from_config(config, ckpt)
        model.bench_config = config
        model.ckpt_path = ckpt
        print(f"Loading done!")
    else:
        print(f"Error: '{public_model}' is not a valid public model.")
        return
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    
    return model