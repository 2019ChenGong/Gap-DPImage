import torch
import random
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset
from data.stylegan3.dataset import ImageFolderDataset
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

import os

os.environ['HF_HOME'] = '/bigtemp/fzv6en/diffuser_cache'

def load_public_model(public_model):  

    if public_model == 'stable-diffusion-2-1-base':
        print(f"Loading model: {public_model}")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")

    elif public_model == 'stable-diffusion-v1-5':
        print(f"Loading model: {public_model}")
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to("cuda")
        model.requires_safety_checker = False
        model.model_id = model_id
        print(f"Loading done!")

    else:
        print(f"Error: '{public_model}' is not a valid public model.")
        return
    
    return model