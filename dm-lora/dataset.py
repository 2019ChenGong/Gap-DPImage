
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset, DatasetDict
import random
from dataset_bench import ImageFolderDataset, get_prompt
import torch
from torch.utils.data import random_split

class RocoDataset(Dataset):
    def __init__(
        self,
        split='train',
        max_images=None,
        size=512,
        center_crop=False,
    ):
        self.ds = load_dataset("eltorio/ROCOv2-radiology")[split]
        self.image_transforms = transforms.Compose(
            [
                # transforms.Pad((int((384-128)/2), 0), fill=0, padding_mode='constant'),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if max_images is not None:
            self.ds = self.ds.select(range(max_images))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        instance_image = self.ds[index]['image'].convert("RGB")
        instance_prompt = self.ds[index]['caption']
        return self.image_transforms(instance_image), instance_prompt


class BenchDataset(Dataset):
    def __init__(
        self,
        name='cifar10_bench',
        path='',
        size=256,
        center_crop=False,
    ):
        self.ds = ImageFolderDataset(path, 3, use_labels=True)
        if "mnist" in name:
            train_size = 55000
        elif "cifar" in name:
            train_size = 45000
        elif "eurosat" in name:
            train_size = 21000
        elif "celeba" in name:
            train_size = 145064
        elif "camelyon" in name:
            train_size = 269538
        elif "covidx" in name:
            train_size = 67863
        elif "octmnist" in name:
            train_size = 97477
        else:
            raise NotImplementedError

        val_size = len(self.ds) - train_size
        torch.manual_seed(0)
        self.ds, _ = random_split(self.ds, [train_size, val_size])
        self.prompt_list = get_prompt(name)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        instance_image, label = self.ds[index]
        instance_image = instance_image.convert("RGB")
        instance_prompt = self.prompt_list[label]
        return self.image_transforms(instance_image), instance_prompt


class PEDESDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and tokenizes prompts.
    Supports manual train/val/test split.
    """

    def __init__(
        self,
        split='train',
        max_images=None,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        size=512,
        center_crop=False,
        seed=42,
    ):
        # 加载原始数据集（假设只有一个 'train' 或默认拆分）
        raw_ds = load_dataset("MaulikMadhavi/CUHK-PEDES-processed")["train"]  # 假设原始只有 train

        # 设置随机种子以确保可复现
        random.seed(seed)
        indices = list(range(len(raw_ds)))
        random.shuffle(indices)

        # 计算划分边界
        total = len(indices)
        train_split = int(train_ratio * total)
        val_split = int((train_ratio + val_ratio) * total)

        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]

        # 构建 DatasetDict
        dataset_dict = DatasetDict({
            'train': raw_ds.select(train_indices),
            'valid': raw_ds.select(val_indices),
            'test': raw_ds.select(test_indices)
        })

        # 选择当前 split
        self.ds = dataset_dict[split]
        self.split = split

        self.image_transforms = transforms.Compose(
            [
                transforms.Pad((int((384-128)/2), 0), fill=0, padding_mode='constant'),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # 可选：限制最大图像数量（用于调试）
        if max_images is not None:
            self.ds = self.ds.select(range(min(max_images, len(self.ds))))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        example = self.ds[index]
        instance_image = example['image']
        instance_prompt = example['text']

        # 如果是训练或验证集，并且 prompt 是列表，随机选一条
        if self.split in ['train', 'valid'] and isinstance(instance_prompt, list):
            instance_prompt = random.choice(instance_prompt)

        example = {}
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms(instance_image)

        return instance_image, instance_prompt

class FashionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and tokenizes prompts.
    Supports manual train/val/test split.
    """

    def __init__(
        self,
        split='train',
        max_images=None,
        train_ratio=0.8,
        size=512,
        center_crop=True,
        seed=42,
    ):
        # 加载原始数据集（假设只有一个 'train' 或默认拆分）
        raw_ds = load_dataset("Marqo/deepfashion-multimodal")["data"]  # 假设原始只有 train

        # 设置随机种子以确保可复现
        random.seed(seed)
        indices = list(range(len(raw_ds)))
        random.shuffle(indices)

        # 计算划分边界
        total = len(indices)
        train_split = int(train_ratio * total)

        train_indices = indices[:train_split]
        test_indices = indices[train_split:]

        # 构建 DatasetDict
        dataset_dict = DatasetDict({
            'train': raw_ds.select(train_indices),
            'test': raw_ds.select(test_indices)
        })

        # 选择当前 split
        self.ds = dataset_dict[split]
        self.split = split

        self.image_transforms = transforms.Compose(
            [
                # transforms.Pad((int((384-128)/2), 0), fill=0, padding_mode='constant'),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # 可选：限制最大图像数量（用于调试）
        if max_images is not None:
            self.ds = self.ds.select(range(min(max_images, len(self.ds))))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        example = self.ds[index]
        instance_image = example['image']
        instance_prompt = example['text']

        example = {}
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms(instance_image)

        return instance_image, instance_prompt

import json
import os
from PIL import Image


class Flickr30kDataset(Dataset):
    """
    自定义 Flickr30k Dataset，使用 Karpathy split (train/val/test)
    支持从本地图像目录和 JSON 划分文件加载数据。
    """

    def __init__(self, image_dir, json_path, size=512,
        center_crop=False, split='train', max_samples=None):
        """
        Args:
            image_dir: 图像文件夹路径，如 'flickr30k/flickr30k_images'
            json_path: dataset_flickr30k.json 的路径
            split: 'train', 'val', 'test'
            transform: 图像变换
            max_samples: 限制最大样本数（用于调试）
        """
        self.image_dir = image_dir
        self.split = split

        # 加载 JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 构建数据列表
        self.data = []
        for img in data['images']:
            if img['split'] == split:
                self.data.append({
                    'image_id': img['imgid'],  # 数字 ID
                    'filename': img['filename'],
                    'text': [sentence['raw'] for sentence in img['sentences']]
                })

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        img_path = os.path.join(self.image_dir, item['filename'])
        
        image = Image.open(img_path).convert("RGB")

        text = item['text']
        # print(text)
        if (self.split in ['train', 'val']):
            text = text[random.randint(0, len(text)-1)]

        return self.image_transforms(image), text

class LoRADataset(Dataset):
    """
    加载由 generator_from_prompt_list 生成的数据集。
    每个样本包含：
        - prompt (str)
        - noft_image (PIL 或 tensor)
        - lora_image (PIL 或 tensor)
    """
    def __init__(
        self,
        dataset_dir: str,
    ):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "images")

        # 加载 prompts.json
        json_path = os.path.join(dataset_dir, "prompts.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"prompts.json not found in {dataset_dir}")

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = random.choice(item["prompt"])

        lora_path = os.path.join(self.image_dir, item["image"])

        lora_img = Image.open(lora_path).convert("RGB")

        return {
            "index": item["index"],
            "text": prompt,
            "image": lora_img
        }
    
from torch.utils.data import IterableDataset

class MarqoDataset(IterableDataset):
    def __init__(
        self,
        split='train',
        processor=None,
        max_images=None,
    ):
        self.processor = processor
        self.ds = load_dataset("Marqo/marqo-GS-10M", split=split, streaming=True)
        self.max_images = max_images

    def __iter__(self):
        ds = self.ds
        if self.max_images is not None:
            ds = ds.take(self.max_images)
        
        for sample in ds:
            image = sample['image']
            prompt = sample['query']

            yield {
            "image": image,
            "text": prompt
        }

    def __len__(self):
        if self.max_images is not None:
            return self.max_images
        raise NotImplementedError("Length unknown for streaming dataset")

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