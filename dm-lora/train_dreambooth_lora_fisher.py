# Note: This file is a modified version of the original file from the diffusers library.

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradientAccumulationPlugin
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import json

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
)

from attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from dataset import PEDESDataset, Flickr30kDataset, RocoDataset, FashionDataset, BenchDataset

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from opacus import PrivacyEngine
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")
logger = get_logger(__name__)


from collections import defaultdict

# def infer_processor_key_from_param_name(param_name):
#     """
#     Convert parameter name like:
#       'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight'
#     to processor key:
#       'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'
#     """
#     if "to_q" in param_name or "to_k" in param_name or "to_v" in param_name or "to_out" in param_name:
#         # Remove trailing ".weight" or ".bias"
#         base = param_name.rsplit(".", 1)[0]  # remove .weight
#         # base is like: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q
#         # Remove the last part (to_q, to_k, etc.)
#         attn_path = ".".join(base.split(".")[:-1])  # ...attn1
#         return f"{attn_path}.processor"
#     return None

# def compute_fisher_for_original_attn(
#     unet,
#     vae,
#     text_encoder,
#     train_dataloader,
#     noise_scheduler,
#     dataloader,
#     accelerator,
#     args,
#     num_batches=20,
#     weight_dtype=torch.float32,
#     variation_weight=False,
# ):
#     unet.eval()
#     fisher = defaultdict(float)

#     # We'll map each attention block to its processor name
#     # e.g., "down_blocks.0.attentions.0" -> ["attn1", "attn2"]
#     # But easier: collect all attention module names that have to_q, etc.
#     progress_bar = tqdm(range(num_batches, args.max_train_steps), disable=not accelerator.is_local_main_process)
#     progress_bar.set_description("Steps")

#     attn_module_names_ = []
#     for name, attn_processor in unet.attn_processors.items():
#         if args.fisher_remove_key is None or args.fisher_remove_key not in name:
#             attn_module_names_.append(name)
#     if args.random_selection:
#         for name in attn_module_names_:
#             fisher[name] = np.random.rand()
#         return fisher
#     attn_module_names = []
#     for name, module in unet.named_modules():
#         if hasattr(module, "to_q") and hasattr(module, "to_k"):
#             if accelerator.is_main_process:
#                 print(name)
#         attn_module_names.append(name)

#     # Prepare everything with our `accelerator`.
#     if args.train_text_encoder:
#         unet, text_encoder, train_dataloader = accelerator.prepare(
#             unet, text_encoder, train_dataloader
#         )
#     else:
#         unet, train_dataloader = accelerator.prepare(
#             unet, train_dataloader
#         )

#     count = 0
#     for step, batch in enumerate(dataloader):
#         if step >= num_batches:
#             break

#         # --- Forward pass (same as your training loop) ---
#         pixel_values = batch["pixel_values"].to(dtype=weight_dtype, device=accelerator.device)
#         if vae is not None:
#             model_input = vae.encode(pixel_values).latent_dist.sample()
#             model_input = model_input * vae.config.scaling_factor
#         else:
#             model_input = pixel_values

#         model_input = model_input.repeat_interleave(args.micro_batch_size, dim=0)
#         noise = torch.randn_like(model_input)
#         bsz = model_input.shape[0]
#         timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()
#         noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

#         if args.pre_compute_text_embeddings:
#             encoder_hidden_states = batch["input_ids"]
#         else:
#             encoder_hidden_states = encode_prompt(
#                 text_encoder,
#                 batch["input_ids"],
#                 batch["attention_mask"],
#                 text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
#             )
#         encoder_hidden_states = encoder_hidden_states.repeat_interleave(args.micro_batch_size, dim=0)

#         if accelerator.unwrap_model(unet).config.in_channels == model_input.shape[1] * 2:
#             noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

#         class_labels = timesteps if args.class_labels_conditioning == "timesteps" else None

#         # --- Compute loss ---
#         unet.zero_grad()
#         model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels).sample

#         if model_pred.shape[1] == 6:
#             model_pred, _ = torch.chunk(model_pred, 2, dim=1)

#         if noise_scheduler.config.prediction_type == "epsilon":
#             target = noise
#         elif noise_scheduler.config.prediction_type == "v_prediction":
#             target = noise_scheduler.get_velocity(model_input, noise, timesteps)
#         else:
#             raise ValueError(...)

#         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
#         accelerator.backward(loss)

#         for name, param in unet.named_parameters():
#             if name.startswith('module.'):
#                 name = name[7:]
#             if param.grad is not None:
#                 # if accelerator.is_main_process:
#                 #     print((param.grad ** 2).sum())
#                 # Check if this param belongs to an attention block
#                 # e.g., "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
#                 if any(attn_name in name for attn_name in attn_module_names):
#                     # Infer the corresponding processor key
#                     proc_key = infer_processor_key_from_param_name(name)
#                     if proc_key:
#                         if proc_key in attn_module_names_:
#                             continue
#                 param.grad = None

#         # accelerator.clip_grad_norm_(unet.parameters(), 1.0)
#         torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=args.fisher_norm)

#         last_name = None

#         # --- Accumulate grad^2 for attention parameters ---
#         for name, param in unet.named_parameters():
#             if name.startswith('module.'):
#                 name = name[7:]
#             # print(name)
#             if param.grad is not None:
#                 # if accelerator.is_main_process:
#                 #     print((param.grad ** 2).sum())
#                 # Check if this param belongs to an attention block
#                 # e.g., "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
#                 if any(attn_name in name for attn_name in attn_module_names):
#                     # Infer the corresponding processor key
#                     print(proc_key)
#                     proc_key = infer_processor_key_from_param_name(name)
#                     if proc_key:
#                         if proc_key in attn_module_names_:
#                             if proc_key in fisher:
#                                 if proc_key == last_name:
#                                     fisher[proc_key][-1] += (param.grad ** 2).sum().item()
#                                 else:
#                                     fisher[proc_key].append((param.grad ** 2).sum().item())
#                             else:
#                                 fisher[proc_key] = [(param.grad ** 2).sum().item()]
#                             last_name = proc_key

#         count += 1
#         progress_bar.update(1)

#     # Normalize
#     keys = list(fisher.keys())
#     for k in fisher:
#         local_tensor = torch.tensor(fisher[k]).to(accelerator.device)
#         all_tensors = accelerator.gather_for_metrics(local_tensor)
#         fisher[k] = all_tensors.cpu().tolist()  # if all_tensors is 1D
#         if accelerator.is_main_process:
#             print(k, np.sum(fisher[k]))
#         mean = np.sum(fisher[k]) + np.random.randn() * args.fisher_sigma * args.fisher_norm

#         if keys.index(k) >= len(keys) // 2:
#             w = args.fisher_weight
#         else:
#             w = 1
#         if variation_weight:
#             std = np.std(fisher[k])
#             if accelerator.is_main_process:
#                 print(k, std)
#             fisher[k] = mean / std
#         else:
#             fisher[k] = mean
#         fisher[k] *= w
#         if accelerator.is_main_process:
#             print(k, fisher[k])
#     return fisher


def infer_fisher_key_from_param_name(param_name):
    """
    Convert parameter name like:
      'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight'
    to fine-grained Fisher key:
      'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q'
    Returns None if not an attention projection param.
    """
    if any(proj in param_name for proj in ["to_q", "to_k", "to_v", "to_out"]):
        # Remove trailing ".weight" or ".bias"
        base = param_name.rsplit(".", 1)[0]  # e.g., ...attn1.to_q
        return base
    return None


def compute_fisher_for_original_attn(
    unet,
    vae,
    text_encoder,
    train_dataloader,
    noise_scheduler,
    dataloader,
    accelerator,
    args,
    num_batches=20,
    weight_dtype=torch.float32,
    variation_weight=False,
):
    unet.eval()
    fisher = defaultdict(float)

    # If random selection, just return random values for all to_q/k/v/o keys
    if args.random_selection:
        # Collect all possible fine-grained keys
        fisher_keys = set()
        for name, _ in unet.named_parameters():
            if name.startswith('module.'):
                name = name[7:]
            key = infer_fisher_key_from_param_name(name)
            if key:
                fisher_keys.add(key)
        for key in fisher_keys:
            fisher[key] = np.random.rand()
        return fisher

    # Prepare model and data
    if args.train_text_encoder:
        unet, text_encoder, train_dataloader = accelerator.prepare(
            unet, text_encoder, train_dataloader
        )
    else:
        unet, train_dataloader = accelerator.prepare(
            unet, train_dataloader
        )

    count = 0
    progress_bar = tqdm(range(num_batches), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Computing Fisher")

    for step, batch in enumerate(dataloader):
        if step >= num_batches:
            break

        # --- Forward pass ---
        pixel_values = batch["pixel_values"].to(dtype=weight_dtype, device=accelerator.device)
        if vae is not None:
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor
        else:
            model_input = pixel_values

        model_input = model_input.repeat_interleave(args.micro_batch_size, dim=0)
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        ).long()
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        if args.pre_compute_text_embeddings:
            encoder_hidden_states = batch["input_ids"]
        else:
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(args.micro_batch_size, dim=0)

        if accelerator.unwrap_model(unet).config.in_channels == model_input.shape[1] * 2:
            noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

        class_labels = timesteps if args.class_labels_conditioning == "timesteps" else None

        # --- Compute loss and backward ---
        unet.zero_grad()
        model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels).sample

        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)

        # Clip gradients (optional but recommended for stability)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=args.fisher_norm)

        # --- Accumulate Fisher info per to_q/k/v/o ---
        for name, param in unet.named_parameters():
            if name.startswith('module.'):
                name = name[7:]
            if param.grad is not None:
                fisher_key = infer_fisher_key_from_param_name(name)
                if fisher_key is not None:
                    # Accumulate squared grad
                    grad_sq = (param.grad ** 2).sum().item()
                    if fisher_key in fisher:
                        fisher[fisher_key] += grad_sq
                    else:
                        fisher[fisher_key] = grad_sq

        count += 1
        progress_bar.update(1)

    # --- Gather across processes and post-process ---
    # Get all keys consistently across devices
    all_keys = list(fisher.keys())
    all_keys_tensor = torch.tensor([hash(k) for k in all_keys], dtype=torch.long, device=accelerator.device)
    gathered_keys = accelerator.gather(all_keys_tensor)
    # But easier: just gather each key's value separately

    final_fisher = {}
    keys_sorted = sorted(fisher.keys())  # Ensure consistent ordering

    for k in keys_sorted:
        local_val = torch.tensor(fisher[k], device=accelerator.device)
        gathered_vals = accelerator.gather(local_val)
        total_val = gathered_vals.sum().cpu().item()  # sum across all processes

        # Add noise if needed (for DP)
        total_val += np.random.randn() * args.fisher_sigma * args.fisher_norm

        # Optional: apply different weights based on position (e.g., first half vs second half)
        w = args.fisher_weight if keys_sorted.index(k) >= len(keys_sorted) // 2 else 1.0

        if variation_weight:
            # Note: we don't have per-batch stats here, so std estimation is not feasible
            # Unless you store per-batch, which complicates things.
            # So skip std normalization unless you modify accumulation to store list.
            # For now, just use total_val.
            final_fisher[k] = total_val * w
        else:
            final_fisher[k] = total_val * w

        if accelerator.is_main_process:
            print(f"{k}: {final_fisher[k]}")

    return final_fisher

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- {'stable-diffusion' if isinstance(pipeline, StableDiffusionPipeline) else 'if'}
- {'stable-diffusion-diffusers' if isinstance(pipeline, StableDiffusionPipeline) else 'if-diffusers'}
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def struct_output(args, accelerator):
    """
    output:
	|- backpack
		|- lora_{r}_{learning_rate}_{learning_rate_text}
			|- *.savetensors
			|- images
				|- *.png
		|- krona_{r}_{learning_rate}_{learning_rate_text}
			|- *.savetensors
			|- images
				|- *.png
	
    |- backpack_dog
        |- lora_{r}_{learning_rate}_{learning_rate_text}
                |- *.savetensors
                |- images
                    |- *.png
        |- krona_{r}_{learning_rate}_{learning_rate_text}
            |- *.savetensors
            |- images
                |- *.png
    """
    print("output folder formatting done.")
    # Check whether output folder exists or not. If not then create
    if accelerator.is_main_process:
        if(os.path.exists(args.output_dir)): pass
        else: os.mkdir(args.output_dir)
    
    
    if(args.adapter_type=="lora"):
        # Now create folder for experiments
        attn_config = ''
        if "k" in args.attn_update_unet: attn_config = attn_config + "k" + str(args.unet_lora_rank_k)
        if "q" in args.attn_update_unet: attn_config = attn_config + "q" + str(args.unet_lora_rank_q)
        if "v" in args.attn_update_unet: attn_config = attn_config + "v" + str(args.unet_lora_rank_v)
        if "o" in args.attn_update_unet: attn_config = attn_config + "o" + str(args.unet_lora_rank_out)
        if(args.unet_tune_mlp): attn_config = attn_config + "f" + str(args.unet_lora_rank_mlp)
        exp = f"lora_{attn_config}_{args.diffusion_model}_{args.learning_rate}"

        if args.attn_keywords != 'attn' and not args.attn_keywords.endswith('json'):
            exp += f'_{args.attn_keywords}'

        if(args.train_text_encoder):
            text_attn_config = ''
            if "k" in args.attn_update_text: text_attn_config = text_attn_config + "k" + str(args.text_lora_rank_k)
            if "q" in args.attn_update_text: text_attn_config = text_attn_config + "q" + str(args.text_lora_rank_q)
            if "v" in args.attn_update_text: text_attn_config = text_attn_config + "v" + str(args.text_lora_rank_v)
            if "o" in args.attn_update_text: text_attn_config = text_attn_config + "o" + str(args.text_lora_rank_out)
            if(args.text_tune_mlp): text_attn_config = text_attn_config + "f" + str(args.text_lora_rank_mlp)
            attn_config = attn_config + "_" + text_attn_config
            exp = f"lora_{attn_config}_{args.diffusion_model}_{args.learning_rate}_{args.learning_rate_text}"
    
    elif(args.adapter_type=="krona"):
        # Now create folder for experiments
        attn_config = ""
        if "k" in args.attn_update_unet: 
            attn_config = attn_config + "k" + str(args.krona_unet_k_rank_a1) + ":" + str(args.krona_unet_k_rank_a2)
        if "q" in args.attn_update_unet: 
            attn_config = attn_config + "q" + str(args.krona_unet_q_rank_a1) + ":" + str(args.krona_unet_q_rank_a2)
        if "v" in args.attn_update_unet: 
            attn_config = attn_config + "v" + str(args.krona_unet_v_rank_a1) + ":" + str(args.krona_unet_v_rank_a2)
        if "o" in args.attn_update_unet: 
            attn_config = attn_config + "o" + str(args.krona_unet_o_rank_a1) + ":" + str(args.krona_unet_o_rank_a2)
        if(args.unet_tune_mlp): 
            attn_config = attn_config + "f" + str(args.krona_unet_ffn_rank_a1) + ":" + str(args.krona_unet_ffn_rank_a2)
        exp = f"krona_{attn_config}_{args.diffusion_model}_{args.learning_rate}"
        
        if(args.train_text_encoder):
            text_attn_config = ''
            if "k" in args.attn_update_text: 
                text_attn_config = text_attn_config + "k" + str(args.krona_text_k_rank_a1) + ":" + str(args.krona_text_k_rank_a2)
            if "q" in args.attn_update_text: 
                text_attn_config = text_attn_config + "q" + str(args.krona_text_q_rank_a1) + ":" + str(args.krona_text_q_rank_a2)
            if "v" in args.attn_update_text: 
                text_attn_config = text_attn_config + "v" + str(args.krona_text_v_rank_a1) + ":" + str(args.krona_text_v_rank_a2)
            if "o" in args.attn_update_text: 
                text_attn_config = text_attn_config + "o" + str(args.krona_text_o_rank_a1) + ":" + str(args.krona_text_o_rank_a2)
            if(args.text_tune_mlp): 
                text_attn_config = text_attn_config + "f" + str(args.krona_text_ffn_rank_a1) + ":" + str(args.krona_text_ffn_rank_a2)    
            attn_config = attn_config + "_" + text_attn_config
            exp = f"krona_{attn_config}_{args.diffusion_model}_{args.learning_rate}_{args.learning_rate_text}"
    else: raise AttributeError(f"{args.adapter_type} wrong adapter format.")

    exp_ = os.path.join(args.output_dir, exp)
    if accelerator.is_main_process:
        if(os.path.exists(exp_)): pass
        else: os.mkdir(exp_)
        os.makedirs(os.path.join(exp_, "samples"), exist_ok=True)

        log_file = os.path.join(exp_, "log.txt")
        if hasattr(logger, "logger"):
            actual_logger = logger.logger
        else:
            actual_logger = logger

        log_file_abs = os.path.abspath(log_file)
        for handler in actual_logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_abs:
                return  # 已存在，安全退出
        # print("0000000000")
        actual_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file_abs, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        actual_logger.addHandler(file_handler)

    return exp_


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--bench_path",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="",
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    
    # our parsers
    parser.add_argument(
        "--unet_lora_rank_k",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => k matrix",
    )
    parser.add_argument(
        "--unet_lora_rank_q",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => q matrix",
    )
    parser.add_argument(
        "--unet_lora_rank_v",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => v matrix",
    )
    parser.add_argument(
        "--unet_lora_rank_out",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => out matrix",
    )
    parser.add_argument("--unet_lora_rank_mlp",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => ffn matrix",
    )
    parser.add_argument(
        "--text_lora_rank_k",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => k matrix",
    )
    parser.add_argument(
        "--text_lora_rank_q",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => q matrix",
    )
    parser.add_argument(
        "--text_lora_rank_v",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => v matrix",
    )
    parser.add_argument(
        "--text_lora_rank_out",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => out matrix",
    )
    parser.add_argument("--text_lora_rank_mlp",
        type=int,
        default=4,
        help="Lora Rank size for matrix decomposition => ffn matrix",
    )
    # krona unet 
    parser.add_argument(
        "--krona_unet_k_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for K attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_k_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for K attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_q_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for Q attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_q_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for Q attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_v_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for V attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_v_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for V attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_o_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for out attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_o_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for out attention matrices in Unet",
    )
    parser.add_argument(
        "--krona_unet_ffn_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for ffn/mlp layers in Unet",
    )
    parser.add_argument(
        "--krona_unet_ffn_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for ffn/mlp layers in Unet",
    )
    # krona text encoder 
    parser.add_argument(
        "--krona_text_k_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for K attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_k_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for K attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_q_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for Q attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_q_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for Q attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_v_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for V attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_v_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for V attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_o_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for out attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_o_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for out attention matrices in Text",
    )
    parser.add_argument(
        "--krona_text_ffn_rank_a2",
        type=int,
        default=16,
        help="KornA Rank size for matrix decomposition for ffn/mlp layers in Text",
    )
    parser.add_argument(
        "--krona_text_ffn_rank_a1",
        type=int,
        default=32,
        help="KornA Rank size for matrix decomposition for ffn/mlp layers in Text",
    )
    
    parser.add_argument(
        "--diffusion_model",
        type=str,
        default="sdxl",
        help="Define the type of diffusion model to be used",
        # choices=["sdxl", "base", "base1-5", "base-2-1"],
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="lora",
        help="Adapter type.",
        choices=["lora", "krona"],
    )

    parser.add_argument(
        "--attn_update_unet",
        type=str,
        default=None,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--fisher_save_name",
        type=str,
        default="tuning_layers.json",
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--fisher_remove_key",
        type=str,
        default=None,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--attn_update_text",
        type=str,
        default=None,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=1500,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--top_k_lora",
        type=float,
        default=0.5,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--fisher_num_batches",
        type=int,
        default=10,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--fisher_sigma",
        type=float,
        default=5,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--fisher_weight",
        type=float,
        default=1,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument(
        "--fisher_norm",
        type=float,
        default=1,
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument("--variation_weight", 
        action="store_true", 
        help="Whether or not to push the model to the GDrive.",
    )

    parser.add_argument("--random_selection", 
        action="store_true", 
        help="Whether or not to push the model to the GDrive.",
    )

    parser.add_argument(
        "--attn_keywords",
        type=str,
        default='attn',
        help="Details about attention matrix (k, q, v, o)",
    )

    parser.add_argument("--delete_and_upload_drive", 
        action="store_true", 
        help="Whether or not to push the model to the GDrive.",
    )
    
    parser.add_argument("--adapter_low_rank", 
        action="store_true",
        help="Whether low rank parameterized format is there or not.",
    )

    parser.add_argument("--unet_tune_mlp",
        action="store_true",
        help="Whether we are finetuning MLP layers as well.",
    )
    parser.add_argument("--text_tune_mlp",
        action="store_true",
        help="Whether we are finetuning MLP layers as well.",
    )
    parser.add_argument(
        "--pretrain_model",
        type=str,
        default=None,
        help="Details about attention matrix (k, q, v, o)",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def unet_ffn_within_attn_processors_state_dict(unet):
    """
    Returns:
        a state dict containing just the ffn processor parameters.
    """
    ffn_processors = unet.ffn_processors 
    ffn_processors_state_dict = {}

    for ffn_processor_key, ffn_processor in ffn_processors.items():
        for parameter_key, parameter in ffn_processor.state_dict().items():
            ffn_processors_state_dict[f"{ffn_processor_key}.{parameter_key}"] = parameter
    return ffn_processors_state_dict


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    r"""
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    plugin = GradientAccumulationPlugin(num_steps=args.gradient_accumulation_steps, sync_with_dataloader=False)
    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_plugin=plugin,
    )

    if accelerator.is_main_process:
        args.output_dir = struct_output(args, accelerator) # structure the output folder

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    for name, param in unet.named_parameters():
        if 'attn' in name:
            param.requires_grad_(True)  # Enable for Fisher computation

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.pretrain_model is not None:
        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(args.pretrain_model)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet)
        unet.fuse_lora()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.instance_prompt != '':
        args.validation_prompt = args.instance_prompt

    def collate_fn(examples, with_prior_preservation=False):
        pixel_values, prompts = zip(*examples)
        if args.instance_prompt != '':
            prompts = [args.instance_prompt] * len(prompts)
        # Tokenize prompts
        text_inputs = tokenize_prompt(tokenizer, prompts, tokenizer_max_length=args.tokenizer_max_length)

        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask
        }

        return batch
    
    train_dataset = BenchDataset(
        name=args.instance_data_dir,
        path=args.bench_path,
        size=args.resolution,
        center_crop=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # ====== NEW: Compute Fisher Info ======
    # if accelerator.is_main_process:
    top_k_lora = args.top_k_lora
    fisher_num_batches = args.fisher_num_batches // accelerator.num_processes
    print("🔍 Computing Fisher information for attention layers...")

    fisher_scores = compute_fisher_for_original_attn(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        train_dataloader=train_dataloader,
        noise_scheduler=noise_scheduler,
        dataloader=train_dataloader,
        accelerator=accelerator,
        args=args,
        num_batches=fisher_num_batches,
        weight_dtype=weight_dtype,
        variation_weight=args.variation_weight,
    )

    # After computing fisher_scores (which now maps '...to_q' -> score)
    if accelerator.is_main_process:
        # Step 1: Sort all fine-grained keys by Fisher score
        sorted_items = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)
        total_num = len(sorted_items)
        top_k_count = int(top_k_lora * total_num)
        top_k_fine_grained = set(name for name, _ in sorted_items[:top_k_count])

        # Step 2: Group by attention module base (e.g., ...attn1)
        selected_attn_modules = defaultdict(list)
        for full_key in top_k_fine_grained:
            # Split into base and projection
            parts = full_key.split(".")
            if any(proj in parts[-1] for proj in ["to_q", "to_k", "to_v", "to_out"]):
                proj_name = parts[-1][3:][:1]
                base_path = ".".join(parts[:-1]) + '.processor'  # e.g., ...attn1
                selected_attn_modules[base_path].append(proj_name)

        # Optional: sort projections for consistent output
        selected_attn_modules = {
            k: sorted(v) for k, v in selected_attn_modules.items()
        }

        # Step 3: Save as structured JSON
        save_path = os.path.join(args.output_dir, args.fisher_save_name)
        with open(save_path, "w") as f:
            json.dump(selected_attn_modules, f, indent=2)

        print(f"✅ Selected top-{top_k_lora * 100:.1f}% ({len(top_k_fine_grained)} / {total_num}) projection layers for LoRA:")
        for base, projs in sorted(selected_attn_modules.items()):
            print(f"  - {base}: [{', '.join(projs)}]")
    # Get top-K layer names
    # if accelerator.is_main_process:
    #     sorted_items = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)
    #     top_k_names = set(name for name, _ in sorted_items[:int(top_k_lora*len(sorted_items))])
    #     with open(os.path.join(args.output_dir, args.fisher_save_name), "w") as f:
    #         json.dump(sorted(top_k_names), f, indent=2)
    #     print(f"✅ Selected top-{top_k_lora} layers for LoRA:")
    #     for name in sorted(top_k_names):
    #         print(f"  - {name}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
