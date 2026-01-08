import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from improved_diffusion import dist_util
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from concurrent.futures import ProcessPoolExecutor
import threading
import math

from .api import API
# from dpsda.pytorch_utils import dev


def _round_to_uint8(image):
    return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)

import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import os

# 确保使用 spawn 启动方式（对 CUDA 必要）
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def _worker_task_group(args):
    """
    Args:
        args: (gpu_id, task_list, pipe_kwargs, common_params)
        common_params: dict with keys: width, height, num_inference_steps, guidance_scale, max_batch_size
    """
    gpu_id, task_list, pipe_kwargs, common_params = args
    device = 'cuda:%d' % gpu_id

    # 加载 pipeline
    pipe = StableDiffusionPipeline.from_pretrained(**pipe_kwargs).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    width = common_params["width"]
    height = common_params["height"]
    num_inference_steps = common_params["num_inference_steps"]
    guidance_scale = common_params["guidance_scale"]
    max_batch_size = common_params["max_batch_size"]

    all_images = []
    all_prompts = []

    for task in task_list:
        prompt, num_samples_for_prompt = task
        images = []
        num_iters = int(np.ceil(num_samples_for_prompt / max_batch_size))
        for i in range(num_iters):
            bs = min(max_batch_size, num_samples_for_prompt - i * max_batch_size)
            with torch.no_grad():
                out = pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=max(bs, 2),
                    output_type='np'
                ).images
            images.append(_round_to_uint8(out[:bs]))
        images = np.concatenate(images, axis=0)
        all_images.append(images)
        all_prompts.extend([prompt] * num_samples_for_prompt)

    torch.cuda.empty_cache()
    del pipe
    return np.concatenate(all_images, axis=0), np.array(all_prompts)

def generate_multigpu(
    prompts,
    num_samples,
    size,
    max_batch_size,
    num_inference_step,
    guidance_scale,
    pipe_kwargs,
    num_gpus=None
):
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA GPU available"

    width, height = list(map(int, size.split('x')))

    # 计算每个 prompt 的样本数
    tasks = []
    for i, prompt in enumerate(prompts):
        n = (num_samples + i) // len(prompts)
        tasks.append((prompt, n))

    # 分组任务到各 GPU
    task_groups = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        task_groups[i % num_gpus].append(task)

    # 准备 common_params（所有 worker 共享的参数）
    common_params = {
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_step,
        "guidance_scale": guidance_scale,
        "max_batch_size": max_batch_size
    }

    # 构建 pool 参数列表：每个元素是 (gpu_id, task_list, pipe_kwargs, common_params)
    pool_args = []
    for gpu_id in range(num_gpus):
        if task_groups[gpu_id]:
            pool_args.append((gpu_id, task_groups[gpu_id], pipe_kwargs, common_params))

    if not pool_args:
        raise ValueError("No tasks to run.")

    # 执行多进程
    with Pool(processes=len(pool_args)) as pool:
        results = list(tqdm(
            pool.imap(_worker_task_group, pool_args),
            total=len(pool_args)
        ))
    # print(results[0][0].shape, results[1][0].shape, results[2][0].shape)
    all_images = np.concatenate([r[0] for r in results], axis=0)
    all_prompts = np.concatenate([r[1] for r in results], axis=0)
    return all_images, all_prompts

def _run_image_variation_on_device(args):
    """
    Run one variation pass on a given GPU.
    Args:
        args: (gpu_id, images_np, prompts, size, variation_degree, pipe_kwargs, common_params)
    """
    gpu_id, images_np, prompts, size, variation_degree, num_variations_per_image, _variation_checkpoint, max_batch_size, num_inference_steps, guidance_scale, = args
    device = 'cuda:%d' % gpu_id

    # Load pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(_variation_checkpoint, torch_dtype=torch.float16).to(device)
    pipe.text_encoder = pipe.text_encoder.to(torch.float16)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    # Preprocess images
    width, height = list(map(int, size.split('x')))
    variation_transform = T.Compose([
        T.Resize((width, height), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    images = [variation_transform(Image.fromarray(im)) for im in images_np]
    images = torch.stack(images)

    variations = []
    num_iters = int(np.ceil(len(images) / max_batch_size))
    for i in range(num_iters):
        start = i * max_batch_size
        end = min((i + 1) * max_batch_size, len(images))
        batch_images = images[start:end]
        batch_prompts = prompts[start:end]

        with torch.no_grad():
            out = pipe(
                prompt=batch_prompts,
                image=batch_images.to(device, dtype=torch.float16),
                num_inference_steps=num_inference_steps,
                strength=variation_degree,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_variations_per_image,
                output_type='np'
            ).images
            out = out.reshape(len(batch_prompts), num_variations_per_image, *out.shape[1:])
        variations.append(out)

    variations = np.concatenate(variations, axis=0)
    torch.cuda.empty_cache()
    del pipe
    return _round_to_uint8(variations)

class StableDiffusionAPI(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_guidance_scale=7.5,
                 random_sampling_num_inference_steps=30,
                 random_sampling_batch_size=10,
                 variation_checkpoint=None,
                 variation_guidance_scale=7.5,
                 variation_num_inference_steps=30,
                 variation_batch_size=10,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_sampling_checkpoint = random_sampling_checkpoint
        self._random_sampling_guidance_scale = random_sampling_guidance_scale
        self._random_sampling_num_inference_steps = \
            random_sampling_num_inference_steps
        self._random_sampling_batch_size = random_sampling_batch_size

        # self._random_sampling_pipe = StableDiffusionPipeline.from_pretrained(
        #     self._random_sampling_checkpoint, torch_dtype=torch.float16, device_map="balanced")
        # self._random_sampling_pipe.safety_checker = None
        # self._random_sampling_pipe = self._random_sampling_pipe.to(dist_util.dev())

        self._variation_checkpoint = variation_checkpoint
        self._variation_guidance_scale = variation_guidance_scale
        self._variation_num_inference_steps = variation_num_inference_steps
        self._variation_batch_size = variation_batch_size

        # self._variation_pipe = \
        #     StableDiffusionImg2ImgPipeline.from_pretrained(
        #         self._variation_checkpoint,
        #         torch_dtype=torch.float16, device_map="balanced")
        # self._variation_pipe.safety_checker = None
        # self._variation_pipe = self._variation_pipe.to(dist_util.dev())

        # self._random_sampling_pipe.set_progress_bar_config(disable=True)
        # self._variation_pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def command_line_parser():
        parser = super(
            StableDiffusionAPI, StableDiffusionAPI).command_line_parser()
        parser.add_argument(
            '--random_sampling_checkpoint',
            type=str,
            required=True,
            help='The path to the checkpoint for random sampling API')
        parser.add_argument(
            '--random_sampling_guidance_scale',
            type=float,
            default=7.5,
            help='The guidance scale for random sampling API')
        parser.add_argument(
            '--random_sampling_num_inference_steps',
            type=int,
            default=50,
            help='The number of diffusion steps for random sampling API')
        parser.add_argument(
            '--random_sampling_batch_size',
            type=int,
            default=10,
            help='The batch size for random sampling API')

        parser.add_argument(
            '--variation_checkpoint',
            type=str,
            required=True,
            help='The path to the checkpoint for variation API')
        parser.add_argument(
            '--variation_guidance_scale',
            type=float,
            default=7.5,
            help='The guidance scale for variation API')
        parser.add_argument(
            '--variation_num_inference_steps',
            type=int,
            default=50,
            help='The number of diffusion steps for variation API')
        parser.add_argument(
            '--variation_batch_size',
            type=int,
            default=10,
            help='The batch size for variation API')
        return parser

    def image_random_sampling(self, num_samples, size, prompts, labels=None):
        """
        Generates a specified number of random image samples based on a given
        prompt and size using OpenAI's Image API.

        Args:
            num_samples (int):
                The number of image samples to generate.
            size (str, optional):
                The size of the generated images in the format
                "widthxheight". Options include "256x256", "512x512", and
                "1024x1024".
            prompts (List[str]):
                The text prompts to generate images from. Each promot will be
                used to generate num_samples/len(prompts) number of samples.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x width x height x
                channels] with type np.uint8 containing the generated image
                samples as numpy arrays.
            numpy.ndarray:
                A numpy array with length num_samples containing prompts for
                each image.
        """
        # max_batch_size = self._random_sampling_batch_size
        # images = []
        # return_prompts = []
        # width, height = list(map(int, size.split('x')))
        # for prompt_i, prompt in enumerate(prompts):
        #     print(prompt)
        #     num_samples_for_prompt = (num_samples + prompt_i) // len(prompts)
        #     num_iterations = int(np.ceil(
        #         float(num_samples_for_prompt) / max_batch_size))
        #     for iteration in tqdm(range(num_iterations)):
        #         batch_size = min(
        #             max_batch_size,
        #             num_samples_for_prompt - iteration * max_batch_size)
        #         images.append(_round_to_uint8(self._random_sampling_pipe(
        #             prompt=prompt,
        #             width=width,
        #             height=height,
        #             disable_progress_bar=True,
        #             num_inference_steps=(
        #                 self._random_sampling_num_inference_steps),
        #             guidance_scale=self._random_sampling_guidance_scale,
        #             num_images_per_prompt=batch_size,
        #             output_type='np').images))
        #     return_prompts.extend([prompt] * num_samples_for_prompt)
        
        # return np.concatensate(images, axis=0), np.array(return_prompts)
    
        pipe_kwargs = {
            "pretrained_model_name_or_path": self._random_sampling_checkpoint,
            "torch_dtype": torch.float16,
        }

        images, prompts = generate_multigpu(
            prompts=prompts,
            num_samples=num_samples,
            size=size,
            max_batch_size=self._random_sampling_batch_size,
            num_inference_step=self._random_sampling_num_inference_steps,
            guidance_scale=self._random_sampling_guidance_scale,
            pipe_kwargs=pipe_kwargs,
            num_gpus=torch.cuda.device_count()
        )
        # print(len(images))

        return images, prompts

    def __image_variation(self, images, additional_info,
                        num_variations_per_image, size, variation_degree):
        """
        Generates a specified number of variations for each image in the input
        array using OpenAI's Image Variation API.

        Args:
            images (numpy.ndarray):
                A numpy array of shape [num_samples x width x height
                x channels] containing the input images as numpy arrays of type
                uint8.
            additional_info (numpy.ndarray):
                A numpy array with the first dimension equaling to
                num_samples containing prompts provided by
                image_random_sampling.
            num_variations_per_image (int):
                The number of variations to generate for each input image.
            size (str):
                The size of the generated image variations in the
                format "widthxheight". Options include "256x256", "512x512",
                and "1024x1024".
            variation_degree (float):
                The image variation degree, between 0~1. A larger value means
                more variation.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x num_variations_per_image
                x width x height x channels] containing the generated image
                variations as numpy arrays of type uint8.
        """
        if not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations = []
        for _ in tqdm(range(num_variations_per_image)):
            sub_variations = self._image_variation(
                images=images,
                prompts=list(additional_info),
                size=size,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)
    
    def image_variation(
        self,
        images,
        additional_info,
        size,
        variation_degree,
        num_variations_per_image,
        num_gpus=None
    ):
        if not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        assert num_gpus > 0

        N = len(images)
        assert len(additional_info) == N

        # 将 images 和 additional_info 按 GPU 数量分片
        chunk_size = math.ceil(N / num_gpus)
        image_chunks = []
        info_chunks = []

        for i in range(num_gpus):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, N)
            if start >= N:
                break  # 防止 GPU 数 > 图像数
            image_chunks.append(images[start:end])
            info_chunks.append(additional_info[start:end])

        actual_num_gpus = len(image_chunks)  # 可能小于 num_gpus（如果 N 很小）

        # 每个 GPU 一个任务：处理自己的 chunk，并生成 num_variations_per_image 个变体
        task_list = []
        for gpu_id in range(actual_num_gpus):
            task_list.append((
                gpu_id,
                image_chunks[gpu_id],          # only a subset of images
                list(info_chunks[gpu_id]),     # corresponding info
                size,
                variation_degree,
                num_variations_per_image,      # <-- now passed to worker
                self._variation_checkpoint,
                self._variation_batch_size,
                self._variation_num_inference_steps,
                self._variation_guidance_scale
            ))

        # 并行执行：每个 GPU 一个线程
        # with ThreadPoolExecutor(max_workers=actual_num_gpus) as executor:
        #     futures = [
        #         executor.submit(_run_image_variation_on_device, task)
        #         for task in task_list
        #     ]
        #     # 按提交顺序收集结果（保序）
        #     chunk_results = []
        #     for future in tqdm(futures, total=len(futures), desc="Generating variations"):
        #         chunk_results.append(future.result())
        
        with ProcessPoolExecutor(max_workers=actual_num_gpus) as executor:
            futures = [executor.submit(_run_image_variation_on_device, task) for task in task_list]
            chunk_results = []
            for future in tqdm(futures, total=len(futures), desc="Generating variations"):
                chunk_results.append(future.result())

        # chunk_results[i] shape: (chunk_size_i, num_variations_per_image, C, H, W)
        # 合并所有 chunk
        final_result = np.concatenate(chunk_results, axis=0)  # (N, num_variations_per_image, C, H, W)
        return final_result
        

        # Run in parallel
        # with Pool(processes=min(num_gpus, num_variations_per_image)) as pool:
        #     results = list(tqdm(
        #         pool.imap(_run_image_variation_on_device, task_list),
        #         total=len(task_list),
        #         desc="Generating variations"
        #     ))

        return np.stack(results, axis=1)

    def _image_variation(self, images, prompts, size, variation_degree):
        width, height = list(map(int, size.split('x')))
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
        images = torch.stack(images).to(dist_util.dev())
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(images.shape[0]) / max_batch_size))
        for iteration in tqdm(range(num_iterations), leave=False):
            variations.append(self._variation_pipe(
                prompt=prompts[iteration * max_batch_size:
                               (iteration + 1) * max_batch_size],
                image=images[iteration * max_batch_size:
                             (iteration + 1) * max_batch_size],
                             disable_progress_bar=True,
                num_inference_steps=self._variation_num_inference_steps,
                strength=variation_degree,
                guidance_scale=self._variation_guidance_scale,
                num_images_per_prompt=1,
                output_type='np').images)
        variations = _round_to_uint8(np.concatenate(variations, axis=0))
        return variations
