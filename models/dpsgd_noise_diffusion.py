import os
import logging
import torch
import copy
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import torchvision
import tqdm
import random

from models.DP_Diffusion.model.ncsnpp import NCSNpp
from models.DP_Diffusion.utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid, compute_fid_with_images
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from models.DP_Diffusion.model.ema import ExponentialMovingAverage
from models.DP_Diffusion.score_losses import EDMLoss, VPSDELoss, VESDELoss, VLoss
from models.DP_Diffusion.denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, VDenoiser
from models.DP_Diffusion.samplers import ddim_sampler, edm_sampler
from models.DP_Diffusion.generate_base import generate_batch, generate_batch_grad

from models.dp_merf import DP_MERF as Freq_Model
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset

from models.DP_Diffusion.rnd import Rnd
from models.DP_MERF.rff_mmd_approx import data_label_embedding, get_rff_mmd_loss, noisy_dataset_embedding
from data.dataset_loader import CentralDataset, random_aug

from models.PE.pe.dp_counter import dp_nn_histogram
from models.PE.apis.improved_diffusion.gaussian_diffusion import create_gaussian_diffusion


import importlib
opacus = importlib.import_module('opacus')

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
import torch
import numpy as np

from models.synthesizer import DPSynther
from PIL import Image


import torch.nn.functional as F

def print_dimensions_and_range(top_x, top_y, global_rank=0):
    """
    Prints the dimensions and value ranges of top_x and top_y tensors.
    
    Args:
        top_x (torch.Tensor): The tensor for images or features.
        top_y (torch.Tensor | np.ndarray): The tensor/array for labels.
    """
    if global_rank != 0:
        return    

    
    # Dimensions and range for top_x
    print("Dimensions of top_x:", top_x.shape)
    print("Value range of top_x: min =", top_x.min().item(), ", max =", top_x.max().item())
    if top_x.dtype in [torch.float32, torch.float64]:
        print("Mean of top_x:", top_x.mean().item())
        print("Standard deviation of top_x:", top_x.std().item())
    
    # Dimensions and range for top_y
    print("\nDimensions of top_y:", top_y.shape)
    if torch.is_tensor(top_y):
        print("Value range of top_y: min =", top_y.min().item(), ", max =", top_y.max().item())
        unique_labels = torch.unique(top_y)
        print("Unique labels in top_y:", unique_labels.tolist())
        if top_y.dtype == torch.int64:
            distribution = torch.bincount(top_y.long())
            print("Label distribution:", distribution.tolist())
    else:
        # numpy array
        print("Value range of top_y: min =", np.min(top_y), ", max =", np.max(top_y))
        unique_labels = np.unique(top_y)
        print("Unique labels in top_y:", unique_labels.tolist())
        if np.issubdtype(top_y.dtype, np.integer):
            distribution = np.bincount(top_y.astype(np.int64))
            print("Label distribution:", distribution.tolist())

def augment_data(images, labels, aug_factor=8, magnitude=9, num_ops=2):
    """
    Augments the images using random augmentation to multiply the dataset size by aug_factor.
    
    Args:
        images (torch.Tensor): Input images tensor of shape [N, C, H, W], values in [0, 1].
        labels (torch.Tensor): Labels tensor of shape [N].
        aug_factor (int): Factor to multiply the dataset size by.
        magnitude (int): Magnitude for RandAugment.
        num_ops (int): Number of operations for RandAugment.
    
    Returns:
        torch.Tensor: Augmented images [N * aug_factor, C, H, W].
        torch.Tensor: Augmented labels [N * aug_factor].
    """
    # 确保输入是torch tensor
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    # 确保images在正确的设备上
    if images.is_cuda:
        images = images.cpu()
    if labels.is_cuda:
        labels = labels.cpu()
    
    N, C, H, W = images.shape
    augmented_images = []
    augmented_labels = []
    
    trans = random_aug(magnitude=magnitude, num_ops=num_ops)
    
    for _ in range(aug_factor):
        aug_batch = []
        for i in range(N):
            # Convert tensor to numpy array (assuming [0, 1] float)
            img_np = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # 确保图像形状正确
            if len(img_np.shape) == 2:  # 灰度图像
                img_np = np.expand_dims(img_np, axis=2)
            
            # Handle grayscale to RGB if C==1
            if C == 1 and img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            
            # 确保图像是2D或3D的
            if len(img_np.shape) == 3 and img_np.shape[2] == 1:
                img_np = img_np.squeeze(axis=2)
            
            # Convert to PIL Image
            if len(img_np.shape) == 2:  # 灰度图像
                pil_img = Image.fromarray(img_np, mode='L')
            else:  # RGB图像
                pil_img = Image.fromarray(img_np, mode='RGB')
            
            # Apply augmentation
            aug_pil = trans(pil_img)
            
            # Convert back to numpy float [0, 1]
            aug_np = np.array(aug_pil).astype(np.float32) / 255.0
            
            # Handle back to grayscale if original C==1
            if C == 1:
                if len(aug_np.shape) == 3:
                    aug_np = np.mean(aug_np, axis=2, keepdims=True)
                else:
                    aug_np = np.expand_dims(aug_np, axis=2)
            
            # Convert to tensor [C, H, W]
            if len(aug_np.shape) == 2:  # 灰度图像
                aug_tensor = torch.from_numpy(aug_np).unsqueeze(0).float()  # [1, H, W]
            else:  # RGB图像
                aug_tensor = torch.from_numpy(aug_np).permute(2, 0, 1).float()  # [C, H, W]
            
            aug_batch.append(aug_tensor)
        
        aug_batch = torch.stack(aug_batch)  # [N, C, H, W]
        augmented_images.append(aug_batch)
        augmented_labels.append(labels.clone())
    
    augmented_images = torch.cat(augmented_images, dim=0)  # [N*aug_factor, C, H, W]
    augmented_labels = torch.cat(augmented_labels, dim=0)  # [N*aug_factor]
    
    return augmented_images, augmented_labels

def compute_loss(features, labels, temperature=0.5):
    """
    计算DDPM的去噪损失和对比学习损失（InfoNCE）的总和。

    参数：
        features (torch.Tensor): DDPM提取的特征，形状 (batch_size, feature_dim)
        labels (torch.Tensor): 二值标签，1表示好图片，0表示坏图片，形状 (batch_size,)
        temperature (float): InfoNCE损失的温度参数，控制相似度分布的锐度

    返回：
        contrastive_loss (torch.Tensor): InfoNCE对比损失
    """
    good_mask = labels == 1  # 选择标签为1的样本

    # InfoNCE对比损失
    batch_size = features.size(0)
    contrastive_loss = torch.tensor(0.0, device=features.device)

    if good_mask.sum() > 1:  # 确保有足够的好图片用于正样本对
        # 提取好图片和坏图片的特征
        good_features = features[good_mask]
        bad_features = features[~good_mask] if (~good_mask).sum() > 0 else None

        # 计算相似度矩阵（好图片之间的相似度）
        good_sim = F.cosine_similarity(
            good_features.unsqueeze(1), good_features.unsqueeze(0), dim=-1
        ) / temperature

        # 正样本对（好图片之间的对角线元素）
        labels_good = torch.arange(good_features.size(0), device=features.device)
        
        # 负样本：好图片与所有其他图片（包括坏图片）
        if bad_features is not None:
            all_sim = F.cosine_similarity(
                good_features.unsqueeze(1), features.unsqueeze(0), dim=-1
            ) / temperature
        else:
            all_sim = good_sim  # 若无坏图片，仅用好图片

        # InfoNCE损失
        contrastive_loss = F.cross_entropy(all_sim, labels_good)


    return contrastive_loss

class PE_Diffusion(DPSynther):
    def __init__(self, config, device, all_config):
        """
        Initializes the model with the provided configuration and device settings.

        Args:
            config (Config): Configuration object containing all necessary parameters.
            device (str): Device to use for computation (e.g., 'cuda:0').
        """
        super().__init__()

        self.local_rank = config.local_rank  # Local rank of the process
        self.global_rank = config.global_rank  # Global rank of the process
        self.global_size = config.global_size  # Total number of processes

        self.denoiser_name = config.denoiser_name  # Name of the denoiser to be used
        self.denoiser_network = config.denoiser_network  # Network architecture for the denoiser
        self.ema_rate = config.ema_rate  # Rate for exponential moving average
        self.network = config.network  # Configuration for the network
        self.sampler = config.sampler  # Sampler configuration
        self.sampler_fid = config.sampler_fid  # FID sampler configuration
        self.sampler_acc = config.sampler_acc  # Accuracy sampler configuration
        self.fid_stats = config.fid_stats  # FID statistics configuration

        self.config = config  # Store the entire configuration
        self.all_config = all_config
        self.device = 'cuda:%d' % self.local_rank  # Set the device based on local rank

        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        label_dim = max(self.private_num_classes, self.public_num_classes)  # Determine the maximum label dimension
        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)
            for k, v in state['model'].items():
                label_dim = v.shape[0]-1
                break
        self.network.label_dim = label_dim  # Set the label dimension for the network
        if 'mode' in self.all_config.pretrain and self.all_config.pretrain.mode != 'time':
            self.freq_model = Freq_Model(self.all_config.model.freq, self.device, self.all_config.train.sigma_sensitivity_ratio)

        # Initialize the denoiser based on the specified name and network
        if self.denoiser_name == 'edm':
            if self.denoiser_network == 'song':
                self.model = EDMDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize EDM denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for EDM denoiser")
        elif self.denoiser_name == 'vpsde':
            if self.denoiser_network == 'song':
                self.model = VPSDEDenoiser(self.config.beta_min, self.config.beta_max - self.config.beta_min,
                                        self.config.scale, NCSNpp(**self.network).to(self.device))  # Initialize VPSDE denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for VPSDE denoiser")
        elif self.denoiser_name == 'vesde':
            if self.denoiser_network == 'song':
                self.model = VESDEDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize VESDE denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for VESDE denoiser")
        elif self.denoiser_name == 'v':
            if self.denoiser_network == 'song':
                self.model = VDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize V denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for V denoiser")
        else:
            raise NotImplementedError("Denoiser name not recognized")

        self.model = self.model.to(self.local_rank)  # Move the model to the specified device
        self.model.train()  # Set the model to training mode
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_rate)  # Initialize EMA for the model parameters

        # Load checkpoint if provided
        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)  # Load the checkpoint
            new_state_dict = {}
            for k, v in state['model'].items():
                new_state_dict[k[7:]] = v  # Adjust the keys to match the model's state dictionary
            logging.info(self.model.load_state_dict(new_state_dict, strict=True))  # Load the state dictionary into the model
            self.ema.load_state_dict(state['ema'])  # Load the EMA state dictionary
            del state, new_state_dict  # Clean up memory

        self.is_pretrain = True  # Flag to indicate pretraining status
        self.generate_noise = torch.randn((config.noise_num//self.global_size, self.network.num_in_channels, self.network.image_size, self.network.image_size))

    def pretrain(self, public_dataloader, config):
        """
        Pre-trains the model using the provided public dataloader and configuration.

        Args:
            public_dataloader (DataLoader): The dataloader for the public dataset.
            config (dict): Configuration dictionary containing various settings and hyperparameters.
        """
        if public_dataloader is None or config.n_epochs == 0:
            # If no public dataloader is provided, set pretraining flag to False and return.
            self.is_pretrain = False
            return
        
        # Set the number of classes in the loss function to the number of private classes.
        config.loss.n_classes = self.private_num_classes
        if config.cond:
            # If conditional training is enabled, set the label unconditioning probability.
            config.loss['label_unconditioning_prob'] = 0.1
        else:
            # If conditional training is disabled, set the label unconditioning probability to 1.0.
            config.loss['label_unconditioning_prob'] = 1.0

        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # Wrap the model with DistributedDataParallel (DDP) for distributed training.
        model = DDP(self.model, device_ids=[self.local_rank])
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and training details if the global rank is 0.
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        # Create a distributed data loader for the public dataset.
        dataset_loader = torch.utils.data.DataLoader(
            dataset=public_dataloader.dataset, 
            batch_size=config.batch_size // self.global_size, 
            sampler=DistributedSampler(public_dataloader.dataset), 
            pin_memory=True, 
            drop_last=True, 
            num_workers=4 if config.batch_size // self.global_size > 8 else 0
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shape of the batches for sampling and FID computation.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, 
                                self.network.image_size, 
                                self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, 
                            self.network.num_in_channels, 
                            self.network.image_size, 
                            self.network.image_size)

        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):
            dataset_loader.sampler.set_epoch(epoch)
            for _, batch in enumerate(dataset_loader):

                if len(batch) == 2:
                    train_x, train_y = batch
                    label = None
                else:
                    train_x, train_y, label = batch

                # Save snapshots and checkpoints at specified intervals.
                if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                    logging.info('Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                            sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes, noise=self.generate_noise)
                        ema.restore(model.parameters())
                    model.train()

                    save_checkpoint(os.path.join(checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                dist.barrier()

                # Compute FID at specified intervals.
                if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes, noise=self.generate_noise)
                        ema.restore(model.parameters())

                        if self.global_rank == 0:
                            logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
                    model.train()
                dist.barrier()

                # Save checkpoints at specified intervals.
                if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                    save_checkpoint(checkpoint_file, state)
                    logging.info('Saving checkpoint at iteration %d' % state['step'])
                dist.barrier()

                # Prepare the input data for training.
                train_x, train_y = train_x.to(self.device) * 2. - 1., train_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, train_x, train_y, noise=self.generate_noise)
                if label is not None:
                    label = label.to(self.device)
                    if self.all_config.train.contrastive == 'v1':
                        loss = (loss * label.float() + loss * (label.float()-1) * self.all_config.train.contrastive_alpha).mean()
                    elif self.all_config.train.contrastive == 'v2':
                        features = model(train_x, torch.ones_like(train_y).float(), train_y, return_feature=True)
                        contrastive_loss = compute_loss(features.reshape(features.shape[0], -1), label)
                        loss = (loss * label.float()).mean() + contrastive_loss * self.all_config.train.contrastive_alpha
                else:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()

                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss.item(), state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))
            torch.cuda.empty_cache()

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()

    def pretrain_freq(self, sensitive_dataloader, config):
        """
        Pre-trains the model using the provided public dataloader and configuration.

        Args:
            public_dataloader (DataLoader): The dataloader for the public dataset.
            config (dict): Configuration dictionary containing various settings and hyperparameters.
        """

        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        self.noise_factor = self.all_config.train.freq.dp.sigma
        logging.info("The noise factor is {}".format(self.noise_factor))
        
        n_data = len(sensitive_dataloader.dataset)  # Number of data points in the sensitive dataset
        rff_sigma = [float(sig) for sig in self.freq_model.rff_sigma.split(',')]
        if self.global_rank == 0:
            _, w_freq = get_rff_mmd_loss(self.freq_model.n_feat, self.freq_model.d_rff, rff_sigma[0], self.local_rank, self.freq_model.private_num_classes, self.noise_factor, sensitive_dataloader.batch_size, self.freq_model.mmd_type)

            noisy_emb = noisy_dataset_embedding(sensitive_dataloader, w_freq, self.freq_model.d_rff, self.local_rank, self.freq_model.private_num_classes, self.noise_factor, self.freq_model.mmd_type, pca_vecs=None, cond=True)
            torch.save({'w_freq': w_freq.w.cpu(), 'noisy_emb': noisy_emb.cpu()}, os.path.join(config.log_dir, 'freq_cache.pth'))

 
        dist.barrier()
        merf_cache = torch.load(os.path.join(config.log_dir, 'freq_cache.pth'))
        w_freq, noisy_emb = merf_cache['w_freq'].to(self.local_rank), merf_cache['noisy_emb'].to(self.local_rank)
        from collections import namedtuple
        rff_param_tuple = namedtuple('rff_params', ['w', 'b'])
        w_freq_param = rff_param_tuple(w=w_freq, b=None)

        def rff_mmd_loss(gen_enc, gen_labels):
            gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq_param, self.freq_model.mmd_type)
            return torch.sum((noisy_emb - gen_emb) ** 2)

        # Wrap the model with DistributedDataParallel (DDP) for distributed training.
        model = DDP(self.model, device_ids=[self.local_rank])
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and training details if the global rank is 0.
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shape of the batches for sampling and FID computation.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, 
                                self.network.image_size, 
                                self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, 
                            self.network.num_in_channels, 
                            self.network.image_size, 
                            self.network.image_size)

        # Training loop over the specified number of epochs.
        n_iter = n_data // config.batch_size
        for epoch in range(config.n_epochs):
            for batch_idx in range(n_iter):
                # Prepare the input data for training.
                gen_samples, gen_y = generate_batch_grad(sampler, (config.batch_size // self.global_size, self.network.num_in_channels, self.network.image_size, self.network.image_size), 
                                            self.device, self.private_num_classes, self.private_num_classes)
                gen_one_hots = torch.nn.functional.one_hot(gen_y, num_classes=self.private_num_classes)
                gen_samples = gen_samples.reshape(config.batch_size // self.global_size, -1)
                loss = rff_mmd_loss(gen_samples, gen_one_hots)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss.item(), state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))
            
            # Save snapshots and checkpoints at specified intervals.
            if self.global_rank == 0:
                logging.info('Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                model.eval()
                with torch.no_grad():
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                        sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes)
                    ema.restore(model.parameters())
                model.train()

                save_checkpoint(os.path.join(checkpoint_dir, 'snapshot_checkpoint.pth'), state)
            dist.barrier()

            # Compute FID at specified intervals.
            model.eval()
            with torch.no_grad():
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes)
                ema.restore(model.parameters())

                if self.global_rank == 0:
                    logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
            model.train()
            dist.barrier()

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()
        
    def warm_up(self, sensitive_train_loader, config):

        time_set = CentralDataset(sensitive_train_loader.dataset, num_classes=self.all_config.sensitive_data.n_classes, **self.all_config.public_data.central)
        time_dataloader = torch.utils.data.DataLoader(dataset=time_set, shuffle=True, drop_last=True, batch_size=self.all_config.pretrain.batch_size_time, num_workers=0)

        if time_set.privacy_history[0] != 0:
            config.dp['privacy_history'] = [time_set.privacy_history]
        if self.global_rank == 0:
            logging.info("Additional privacy cost: {}".format(str(time_set.privacy_history)))
        if 'auxiliary' not in self.all_config.pretrain.mode and self.all_config.pretrain.mode != 'time':
            if self.global_rank == 0:
                # freq_model = Freq_Model(self.all_config.model.merf, self.device, self.all_config.train.sigma_sensitivity_ratio)
                self.freq_model.train(sensitive_train_loader, self.all_config.train.freq)
                syn_data, syn_labels = self.freq_model.generate(self.all_config.gen.freq)
            dist.barrier()

            syn = np.load(os.path.join(self.all_config.gen.freq.log_dir, 'gen.npz'))
            syn_data, syn_labels = syn["x"], syn["y"]
            freq_train_set = TensorDataset(torch.from_numpy(syn_data).float(), torch.from_numpy(syn_labels).long())
            freq_train_loader = DataLoader(dataset=freq_train_set, shuffle=True, drop_last=True, batch_size=self.all_config.pretrain.batch_size_freq, num_workers=16)

        if self.all_config.pretrain.mode != 'time':
            config.dp['privacy_history'].append([self.all_config.train.freq.dp.sigma, 1, 1])

        # self.freq_train_loader = freq_train_loader
        # self.time_dataloader = time_dataloader
        # return config
    
        if self.all_config.pretrain.mode == 'freq_time':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_freq'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            self.pretrain(freq_train_loader, self.all_config.pretrain)
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir[:-5] + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain)
        elif self.all_config.pretrain.mode == 'time_freq':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain)
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir[:-5] + '_freq'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            self.pretrain(freq_train_loader, self.all_config.pretrain)
        elif self.all_config.pretrain.mode == 'freq':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_freq'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            self.pretrain(freq_train_loader, self.all_config.pretrain)
        elif self.all_config.pretrain.mode == 'time':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain)
        elif self.all_config.pretrain.mode == 'mix':
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            pretrain_set = ConcatDataset([freq_train_set, self.time_dataloader.dataset])
            self.pretrain(DataLoader(dataset=pretrain_set, shuffle=True, drop_last=True, batch_size=self.all_config.pretrain.batch_size, num_workers=16), self.all_config.pretrain)
        else:
            raise NotImplementedError
        
        self.freq_train_loader = freq_train_loader
        self.time_dataloader = time_dataloader
        torch.cuda.empty_cache()
        return config

    def pe_pretrain(self, public_dataloader, config, start_optimizer=None):
        if public_dataloader is None or config.n_epochs == 0:
            # If no public dataloader is provided, set pretraining flag to False and return.
            self.is_pretrain = False
            return
        
        # Set the number of classes in the loss function to the number of private classes.
        config.loss.n_classes = self.private_num_classes
        if config.cond:
            # If conditional training is enabled, set the label unconditioning probability.
            config.loss['label_unconditioning_prob'] = 0.1
        else:
            # If conditional training is disabled, set the label unconditioning probability to 1.0.
            config.loss['label_unconditioning_prob'] = 1.0

        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # Wrap the model with DistributedDataParallel (DDP) for distributed training.
        model = DDP(self.model, device_ids=[self.local_rank])
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")
        
        if start_optimizer is not None:
            optimizer.load_state_dict(start_optimizer.state_dict())

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and training details if the global rank is 0.
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        # Create a distributed data loader for the public dataset.
        dataset_loader = torch.utils.data.DataLoader(
            dataset=public_dataloader.dataset, 
            batch_size=config.batch_size // self.global_size, 
            sampler=DistributedSampler(public_dataloader.dataset), 
            pin_memory=True, 
            drop_last=True, 
            num_workers=4 if config.batch_size // self.global_size > 8 else 0
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shape of the batches for sampling and FID computation.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, 
                                self.network.image_size, 
                                self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, 
                            self.network.num_in_channels, 
                            self.network.image_size, 
                            self.network.image_size)

        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):
            dataset_loader.sampler.set_epoch(epoch)
            for _, batch in enumerate(dataset_loader):

                if len(batch) == 2:
                    train_x, train_y = batch
                    label = None
                else:
                    train_x, train_y, label = batch

                # Save snapshots and checkpoints at specified intervals.
                if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                    logging.info('Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                            sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes, noise=self.generate_noise)
                        ema.restore(model.parameters())
                    model.train()

                    save_checkpoint(os.path.join(checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                dist.barrier()

                # Compute FID at specified intervals.
                # if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                #     model.eval()
                #     with torch.no_grad():
                #         ema.store(model.parameters())
                #         ema.copy_to(model.parameters())
                #         fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes)
                #         ema.restore(model.parameters())

                #         if self.global_rank == 0:
                #             logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
                #     model.train()
                # dist.barrier()

                # Compute FID at each epoch.

                # Save checkpoints at specified intervals.
                if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                    save_checkpoint(checkpoint_file, state)
                    logging.info('Saving checkpoint at iteration %d' % state['step'])
                dist.barrier()

                # Prepare the input data for training.
                train_x, train_y = train_x.to(self.device) * 2. - 1., train_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, train_x, train_y, noise=self.generate_noise)
                if label is not None:
                    label = label.to(self.device)
                    if self.all_config.train.contrastive == 'v1':
                        loss = (loss * label.float() + loss * (label.float()-1) * self.all_config.train.contrastive_alpha).mean()
                    elif self.all_config.train.contrastive == 'v2':
                        features = model(train_x, torch.ones_like(train_y).float(), train_y, return_feature=True)
                        contrastive_loss = compute_loss(features.reshape(features.shape[0], -1), label)
                        loss = (loss * label.float()).mean() + contrastive_loss * self.all_config.train.contrastive_alpha
                else:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()

                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss.item(), state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())

            # Compute FID at each epoch.
            model.eval()
            with torch.no_grad():
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes, noise=self.generate_noise)
                ema.restore(model.parameters())

                if self.global_rank == 0:
                    logging.info('FID at epoch %d: %.6f' % (epoch + 1, fid))
            model.train()
            dist.barrier()
            
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))
            torch.cuda.empty_cache()

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()

    def pe_vote(self, images_to_selected, labels_to_selected, image_categories, sensitive_features, sensitive_labels, config, sigma=5, num_nearest_neighbor=1, nn_mode='L2', count_threshold=4.0, selection_ratio=0.1, device=None, sampler=None):
        features_to_selected = []
        batch_size = 100

        dataset = TensorDataset(images_to_selected)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            images_batch = batch[0]
            with torch.no_grad():
                if images_batch.shape[1] == 1:
                    images_batch = images_batch.repeat(1, 3, 1, 1)
                features_batch = self.inception_model.get_feature_batch(images_batch.to(device))
                features_batch = features_batch.detach().cpu().numpy() 
                features_to_selected.append(features_batch)

        features_to_selected = np.concatenate(features_to_selected, axis=0)

        top_indices = []
        bottom_indices = []
        for class_i in range(self.private_num_classes):
            sub_count, sub_clean_count = dp_nn_histogram(
                public_features=features_to_selected[labels_to_selected == class_i],
                private_features=sensitive_features[sensitive_labels == class_i],
                noise_multiplier=sigma,
                num_nearest_neighbor=num_nearest_neighbor,
                mode=nn_mode,
                threshold=count_threshold,
                device=self.local_rank,
                verbose=False,
            )

            class_mask = (labels_to_selected == class_i)
            class_indices = np.where(class_mask)[0]
            sorted_idx_within_class = np.argsort(sub_count)[::-1]
            n_select = max(1, int(selection_ratio * len(class_indices)))
            
            top_indices_within_class = sorted_idx_within_class[:n_select]
            selected_global_indices = class_indices[top_indices_within_class]
            top_indices.append(selected_global_indices)

            bottom_indices_within_class = sorted_idx_within_class[-n_select:]
            selected_bad_indices = class_indices[bottom_indices_within_class]
            bottom_indices.append(selected_bad_indices)

        top_indices = np.concatenate(top_indices)
        if self.global_rank == 0:
            logging.info(f"Before unique - top_indices length: {len(top_indices)}")

        top_indices = np.unique(top_indices)
        if self.global_rank == 0:
            logging.info(f"After unique - top_indices length: {len(top_indices)}")
        top_images = images_to_selected[top_indices]
        top_labels = labels_to_selected[top_indices]
        if image_categories is not None:
            top_categories = image_categories[top_indices]

        bottom_indices = np.concatenate(bottom_indices)
        if self.global_rank == 0:
            logging.info(f"Before unique - bottom_indices length: {len(bottom_indices)}")
        # 去重，避免重复选择同一个样本
        bottom_indices = np.unique(bottom_indices)
        if self.global_rank == 0:
            logging.info(f"After unique - bottom_indices length: {len(bottom_indices)}")
        bottom_images = images_to_selected[bottom_indices]
        bottom_labels = labels_to_selected[bottom_indices]
        if image_categories is not None:
            bottom_categories = image_categories[bottom_indices]
        if self.global_rank == 0 and image_categories is not None:
            logging.info(f"Selected top results:\nFreq: {np.sum(top_categories==0)} / {np.sum(image_categories==0)} Time: {np.sum(top_categories==1)} / {np.sum(image_categories==1)} Gen: {np.sum(top_categories==2)} / {np.sum(image_categories==2)}")
            logging.info(f"Selected bottom results:\nFreq: {np.sum(bottom_categories==0)} / {np.sum(image_categories==0)} Time: {np.sum(bottom_categories==1)} / {np.sum(image_categories==1)} Gen: {np.sum(bottom_categories==2)} / {np.sum(image_categories==2)}")

        if sampler is not None:
            top_images = self._image_variation(top_images, top_labels, sampler=sampler)
            bottom_images = self._image_variation(bottom_images, bottom_labels, sampler=sampler)
        torch.cuda.empty_cache()

        return top_images, top_labels, bottom_images, bottom_labels

    def constractive_learning(self, top_x, top_y, bottom_x, bottom_y, epoch, config):

        if self.global_rank == 0:
            logging.info(f"Starting constractive_learning with top_x shape: {top_x.shape}")
        
        # Step 1: Generate variants of top_x to reach model.noise_num
        current_top_count = len(top_x)
        target_count = self.all_config.model.noise_num
        variants_needed = max(0, target_count - current_top_count)
        
        if variants_needed > 0:
            if self.global_rank == 0:
                logging.info(f"Generating {variants_needed} variants using _image_variation")
            
            # Generate variants using _image_variation
            # We need to repeat top_x to get enough samples for variation
            repeat_factor = (variants_needed // current_top_count) + 1
            repeated_top_x = np.tile(top_x, (repeat_factor, 1, 1, 1))
            repeated_top_y = np.tile(top_y, (repeat_factor,))
            
            # Create a sampler function for _image_variation
            # This sampler will use the diffusion model to generate variations
            def variation_sampler(x, y, start_sigma=None):
                # Convert to the right format for the diffusion model
                x = x * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
                
                # Use the diffusion model to generate variations
                # We'll use a simple approach: add noise and then denoise
                with torch.no_grad():
                    # Add some noise to create variation
                    # Use config parameter to control noise strength
                    variant_noise_scale = getattr(config, 'variant_noise', 0.1)
                    if self.global_rank == 0:
                        logging.info(f"Using variant_noise_scale: {variant_noise_scale}")
                    noise = torch.randn_like(x) * variant_noise_scale
                    noisy_x = x + noise
                    
                    # Use the diffusion model to denoise (this creates a variation)
                    # For simplicity, we'll just return the noisy version
                    # In practice, you might want to use the actual diffusion sampling process
                    return noisy_x
            
            # Use _image_variation to generate variants
            variant_images = self._image_variation(
                torch.from_numpy(repeated_top_x).float(), 
                repeated_top_y, 
                variation_degree=0.1, 
                sampler=variation_sampler, 
                batch_size=100
            )
            
            # Convert back to numpy and select the needed amount
            variant_images = variant_images.numpy()
            variant_labels = repeated_top_y[:len(variant_images)]
            
            # Select exactly the number we need
            variant_images = variant_images[:variants_needed]
            variant_labels = variant_labels[:variants_needed]
            
            # Combine original top_x with variants
            combined_top_x = np.concatenate([top_x, variant_images], axis=0)
            combined_top_y = np.concatenate([top_y, variant_labels], axis=0)
            
            if self.global_rank == 0:
                logging.info(f"Combined top images shape: {combined_top_x.shape}")
        else:
            combined_top_x = top_x
            combined_top_y = top_y
        
        # Step 2: Convert images to initial noise using diffusion forward process
        if self.global_rank == 0:
            logging.info("Converting images to initial noise using diffusion forward process")
        
        # Convert images to the right format (0-1 range to -1 to 1 range)
        images_tensor = torch.from_numpy(combined_top_x).float()
        images_tensor = images_tensor * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
        images_tensor = images_tensor.to(self.device)
        
        # Generate initial noise by adding noise at maximum timestep
        max_timestep = self._diffusion.num_timesteps - 1
        t_tensor = torch.full((len(images_tensor),), max_timestep, device=self.device, dtype=torch.long)
        
        # Generate random noise
        noise = torch.randn_like(images_tensor)
        
        # Apply forward diffusion process to get noisy images at max timestep
        noisy_images = self._diffusion.q_sample(images_tensor, t_tensor, noise=noise)
        
        # The noisy images at max timestep are essentially the "initial noise" we want
        initial_noise = noisy_images.detach().cpu()
        
        # Step 3: Replace the original generate_noise with our computed initial noise
        if self.global_rank == 0:
            logging.info(f"Replacing generate_noise with computed initial noise, shape: {initial_noise.shape}")
        
        # Ensure we have the right shape
        if initial_noise.shape[0] >= target_count:
            self.generate_noise = initial_noise[:target_count]
        else:
            # If we don't have enough, pad with random noise
            padding_needed = target_count - initial_noise.shape[0]
            padding_noise = torch.randn(padding_needed, *initial_noise.shape[1:])
            self.generate_noise = torch.cat([initial_noise, padding_noise], dim=0)
        
        if self.global_rank == 0:
            logging.info(f"Final generate_noise shape: {self.generate_noise.shape}")
            logging.info("Completed - noise replacement done")

        torch.cuda.empty_cache()


    def train(self, sensitive_dataloader, config):
        """
        Trains the model using the provided sensitive data loader and configuration.

        Args:
            sensitive_dataloader (DataLoader): DataLoader containing the sensitive data.
            config (Config): Configuration object containing various settings for training.
        """
        
        if sensitive_dataloader is None or config.n_epochs == 0:
            return

        if 'mode' in self.all_config.pretrain:
            config = self.warm_up(sensitive_dataloader, config)
        
        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank
        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes

        # Define directories for saving samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if this is the main process.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
        
        if config.partly_finetune:
            # If partial fine-tuning is enabled, freeze certain layers.
            trainable_parameters = []
            for name, param in self.model.named_parameters():
                layer_idx = int(name.split('.')[2])
                if layer_idx > 3 and 'NIN' not in name:
                    param.requires_grad = False
                    if self.global_rank == 0:
                        logging.info('{} is frozen'.format(name))
                else:
                    trainable_parameters.append(param)
        else:
            # Otherwise, all parameters are trainable.
            trainable_parameters = self.model.parameters()

        # Wrap the model with DPDDP for distributed training with differential privacy.
        model = DPDDP(self.model)
        # Initialize Exponential Moving Average (EMA) for model parameters.
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the state dictionary to keep track of the training process.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and other training details.
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])

        # Initialize the Privacy Engine for differential privacy.
        privacy_engine = PrivacyEngine()
        if 'privacy_history' in config.dp and config.dp.privacy_history is not None:
            account_history = [tuple(item) for item in config.dp.privacy_history]
        else:
            account_history = None

        # Make the model, optimizer, and data loader private.
        model, optimizer, dataset_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=sensitive_dataloader,
            target_delta=config.dp.delta,
            target_epsilon=config.dp.epsilon,
            epochs=config.n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=config.loss.n_noise_samples,
            account_history=account_history,
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function not supported")

        # Initialize the Inception model for feature extraction.
        self.inception_model = InceptionFeatureExtractor()
        self.inception_model.model = self.inception_model.model.to(self.device)

        pe_freq = config.pe_freq
        freq_images = []
        freq_labels = []
        freq_features = []
        time_images = []
        time_labels = []
        time_features = []
        for x, y in self.freq_train_loader:
            freq_images.append(x)
            freq_labels.append(y)

        for x, y in self.time_dataloader:
            time_images.append(x)
            time_labels.append(y)

        freq_images = torch.cat(freq_images)[:500]
        freq_labels = torch.cat(freq_labels)[:500]
        time_images = torch.cat(time_images)
        time_labels = torch.cat(time_labels)

        sensitive_features = []
        sensitive_labels = []
        sensitive_dataloader_inc = DataLoader(sensitive_dataloader.dataset, batch_size=100)
        for x, y in sensitive_dataloader_inc:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            features_batch = self.inception_model.get_feature_batch(x.to(self.device))
            sensitive_features.append(features_batch.detach().cpu())
            sensitive_labels.append(y)
        self.sensitive_features = torch.cat(sensitive_features)
        self.sensitive_labels = torch.cat(sensitive_labels)

        # Define the sampler function for generating images.
        def sampler(x, y=None, start_sigma=None, start_t=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, start_sigma=start_sigma, start_t=start_t, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")
        self._diffusion = create_gaussian_diffusion(
            steps=1000,
            learn_sigma=True,
            noise_schedule="cosine",
            timestep_respacing=str(self.sampler.num_steps),)
        self._sigma_list = self._diffusion.sqrt_one_minus_alphas_cumprod / self._diffusion.sqrt_alphas_cumprod

        # Define the shapes for sampling images.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, self.network.image_size, self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, self.network.num_in_channels,
                            self.network.image_size, self.network.image_size)

        # Start the training loop.
        for epoch in range(config.n_epochs):
            pe_lock = True
            
            # PE training at epoch level (not batch level)
            if epoch in pe_freq and pe_lock:
                # PE training
                """
                Key hyper-parameter:
                1. voting epoch interval; 2. voting privacy budget; 3. synthetic image size; 
                4. mean and frequency image size; 5. top and bottom image size;
                """
                if self.global_rank == 0:
                    logging.info("PE training start!")

                # 确保所有进程使用相同的随机种子来生成样本
                torch.manual_seed(42)
                np.random.seed(42)
                
                gen_x = []
                gen_y = []
                pe_batch_size = 500
                for _ in range(config.contrastive_num_samples//self.global_size//pe_batch_size+1):
                    gen_x_i, gen_y_i = generate_batch(sampler, (pe_batch_size, self.network.num_in_channels, self.network.image_size, self.network.image_size), self.device, self.private_num_classes, self.private_num_classes, noise=self.generate_noise)
                    gen_x.append(gen_x_i)
                    gen_y.append(gen_y_i)
                gen_x = torch.cat(gen_x)[:config.contrastive_num_samples]
                gen_y = torch.cat(gen_y)[:config.contrastive_num_samples]

                # combine
                images_to_select = torch.cat([freq_images, time_images, gen_x.detach().cpu()])
                label_to_select = torch.cat([freq_labels, time_labels, gen_y.detach().cpu()])
                image_categories = torch.tensor([0]*len(freq_images)+[1]*len(time_images)+[2]*len(gen_x)).long()

                fid_before_selection = compute_fid_with_images(images_to_select, fid_sampling_shape, self.inception_model, self.fid_stats, self.device)
                
                pe_variation_enabled = hasattr(config, 'pe_variation') and config.pe_variation
                sampler_to_use = sampler if pe_variation_enabled else None
                if self.global_rank == 0:
                    logging.info(f"pe_variation enabled: {pe_variation_enabled}, sampler_to_use: {sampler_to_use is not None}")
                
                torch.manual_seed(42)
                np.random.seed(42)
                
                # 所有进程都执行pe_vote，但由于相同的随机种子，结果应该一致
                top_x, top_y, bottem_x, bottem_y = self.pe_vote(images_to_select, label_to_select.numpy(), image_categories.numpy(), self.sensitive_features.numpy(), self.sensitive_labels.numpy(), selection_ratio=config.contrastive_selection_ratio, config=self.all_config, device=self.device, sampler=sampler_to_use)

                print_dimensions_and_range(top_x, top_y, self.global_rank)

                # Data augmentation control
                argu_enabled = getattr(config, 'argu', False)
                if self.global_rank == 0:
                    logging.info(f"Data augmentation enabled: {argu_enabled}")
                
                if argu_enabled:
                    top_x, top_y = augment_data(top_x, top_y, aug_factor=8, magnitude=9, num_ops=2)

                print_dimensions_and_range(top_x, top_y, self.global_rank)
                
                fid_top = compute_fid_with_images(top_x, fid_sampling_shape, self.inception_model, self.fid_stats, self.device)
                fid_bottom = compute_fid_with_images(bottem_x, fid_sampling_shape, self.inception_model, self.fid_stats, self.device)

                if self.global_rank == 0:
                    logging.info("PE Selecting end!")
                    logging.info(f"fid_before_selection: {fid_before_selection} fid_top: {fid_top} fid_bottom: {fid_bottom}")

                # constractiving learning
                torch.cuda.empty_cache()
                self.constractive_learning(top_x, top_y, bottem_x, bottem_y, epoch, config)

                if self.global_rank == 0:
                    indices = torch.randperm(images_to_select.size(0))
                    images_to_select = images_to_select[indices]
                    label_to_select = label_to_select[indices]

                    # 修复NumPy数组的索引问题
                    indices = torch.randperm(top_x.size(0))
                    top_x = top_x[indices]
                    top_y = top_y[indices.cpu().numpy()]  # 转换为numpy索引

                    indices = torch.randperm(bottem_x.size(0))
                    bottem_x = bottem_x[indices]
                    bottem_y = bottem_y[indices.cpu().numpy()]  # 转换为numpy索引

                    show_images = []
                    for cls in range(self.private_num_classes):
                        show_images.append(images_to_select[label_to_select==cls][:8])
                    show_images = np.concatenate(show_images)
                    torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(self.all_config.pretrain.log_dir, "samples", 'no_selected_samples.png'), padding=1, nrow=8)

                    show_images = []
                    for cls in range(self.private_num_classes):
                        
                        mask = (top_y == cls)
                        if torch.any(mask) if isinstance(mask, torch.Tensor) else np.any(mask):
                            selected_images = top_x[mask][:8]

                            if isinstance(selected_images, torch.Tensor):
                                selected_images = selected_images.cpu().numpy()
                            show_images.append(selected_images)
                    if show_images:
                        show_images = np.concatenate(show_images)
                        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(self.all_config.pretrain.log_dir, "samples", 'top_sample.png'), padding=1, nrow=8)

                    show_images = []
                    for cls in range(self.private_num_classes):
                        
                        mask = (bottem_y == cls)
                        if torch.any(mask) if isinstance(mask, torch.Tensor) else np.any(mask):
                            selected_images = bottem_x[mask][:8]
                            
                            if isinstance(selected_images, torch.Tensor):
                                selected_images = selected_images.cpu().numpy()
                            show_images.append(selected_images)
                    if show_images:
                        show_images = np.concatenate(show_images)
                        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(self.all_config.pretrain.log_dir, "samples", 'bottom_sample.png'), padding=1, nrow=8)
                    logging.info("PE training end!")
                pe_lock = False
            
            with BatchMemoryManager(
                    data_loader=dataset_loader,
                    max_physical_batch_size=config.dp.max_physical_batch_size,
                    optimizer=optimizer,
                    n_splits=config.n_splits if config.n_splits > 0 else None) as memory_safe_data_loader:

                for local_step, (train_x, train_y) in enumerate(memory_safe_data_loader):
                    if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                        # Save a snapshot checkpoint and sample a batch of images.
                        logging.info(
                            'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                                sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes, noise=self.generate_noise)
                            ema.restore(model.parameters())
                        model.train()

                        save_checkpoint(os.path.join(
                            checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                    dist.barrier()

                    if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                        # Compute FID score and log it.
                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, self.inception_model, self.fid_stats, self.device, self.private_num_classes, noise=self.generate_noise)
                            ema.restore(model.parameters())

                            if self.global_rank == 0:
                                logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
                            dist.barrier()
                        model.train()

                    if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                        # Save a checkpoint at regular intervals.
                        checkpoint_file = os.path.join(
                            checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                        save_checkpoint(checkpoint_file, state)
                        logging.info(
                            'Saving checkpoint at iteration %d' % state['step'])
                    dist.barrier()
                    

                    if len(train_y.shape) == 2:
                        # Preprocess the input data.
                        train_x = train_x.to(torch.float32) / 255.
                        train_y = torch.argmax(train_y, dim=1)
                    
                    x = train_x.to(self.device) * 2. - 1.
                    y = train_y.to(self.device).long()

                    # Perform a forward pass and backpropagation.
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean(loss_fn(model, x, y, noise=self.generate_noise))
                    loss.backward()
                    optimizer.step()

                    if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                        # Log the loss at regular intervals.
                        logging.info('Loss: %.4f, step: %d' %
                                    (loss.item(), state['step'] + 1))
                    dist.barrier()

                    state['step'] += 1
                    if not optimizer._is_last_step_skipped:
                        state['ema'].update(model.parameters())

                # Log the epsilon value after each epoch.
                logging.info('Eps-value after %d epochs: %.4f' %
                            (epoch + 1, privacy_engine.get_epsilon(config.dp.delta)))

        if self.global_rank == 0:
            # Save the final checkpoint.
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        self.ema = ema

    def _image_variation(self, images, labels, variation_degree=0.1, sampler=None, batch_size=100):
        if self.global_rank == 0:
            logging.info(f"_image_variation input - images shape: {images.shape}, labels shape: {labels.shape}")
        
        samples = []
        images_list = torch.split(images, split_size_or_sections=batch_size)
        labels_list = torch.split(torch.from_numpy(labels).long(), split_size_or_sections=batch_size)
        
        if self.global_rank == 0:
            logging.info(f"Split into {len(images_list)} batches, batch sizes: {[batch.size(0) for batch in images_list]}")
        
        # 确保处理所有批次，包括最后一个不完整的批次
        for i in range(len(images_list)):
            if images_list[i].size(0) == 0:  # 跳过空批次
                if self.global_rank == 0:
                    logging.info(f"Skipping empty batch {i}")
                continue
            with torch.no_grad():
                sigma_idx = int(len(self._sigma_list) * (1 - variation_degree))
                sigma_idx = max(0, min(sigma_idx, len(self._sigma_list) - 1))
                x = sampler(images_list[i].to(self.device), labels_list[i].to(self.device), start_sigma=self._sigma_list[sigma_idx])
            samples.append(x.detach().cpu())
        
        if samples:  # 确保有样本才进行拼接
            samples = torch.cat(samples).clamp(-1., 1.)
            result = (samples + 1) / 2
            if self.global_rank == 0:
                logging.info(f"_image_variation output - result shape: {result.shape}")
            return result
        else:
            if self.global_rank == 0:
                logging.info("No samples processed, returning original images")
            return images  # 如果没有样本，返回原始图像

    def generate(self, config, sampler_config=None):

        logging.info("start to generate {} samples".format(config.data_num))
        
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        
        # Synchronize all processes
        dist.barrier()

        # Define the shape for the sampling batch
        sampling_shape = (config.batch_size, self.network.num_in_channels, self.network.image_size, self.network.image_size)

        # Wrap the model with DistributedDataParallel for multi-GPU training
        model = DDP(self.model)
        model.eval()  # Set the model to evaluation mode
        
        # Copy the exponential moving average parameters to the model
        self.ema.copy_to(model.parameters())

        # Define a function to handle different types of samplers
        sampler_config = self.sampler_acc if sampler_config is None else sampler_config
        def sampler_acc(x, y=None):
            if sampler_config.type == 'ddim':
                return ddim_sampler(x, y, model, **sampler_config)
            elif sampler_config.type == 'edm':
                return edm_sampler(x, y, model, **sampler_config)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Initialize lists to store synthetic data and labels if this is the main process
        if self.global_rank == 0:
            syn_data = []
            syn_labels = []

        if 'pe_last' in config and config['pe_last']:
            config.data_num = int(config.data_num / config.selection_ratio)

        # Loop to generate the required number of samples
        for _ in range(config.data_num // (sampling_shape[0] * self.global_size) + 1):
            # Generate a batch of samples and labels
            x, y = generate_batch(sampler_acc, sampling_shape, self.device, self.private_num_classes, self.private_num_classes, noise=self.generate_noise)
            
            # Synchronize all processes
            dist.barrier()
            
            # Prepare tensors for gathering results from all processes
            if self.global_rank == 0:
                gather_x = [torch.zeros_like(x) for _ in range(self.global_size)]
                gather_y = [torch.zeros_like(y) for _ in range(self.global_size)]
            else:
                gather_x = None
                gather_y = None
            
            # Gather the generated samples and labels from all processes
            dist.gather(x, gather_x)
            dist.gather(y, gather_y)
            
            # If this is the main process, collect the gathered data
            if self.global_rank == 0:
                syn_data.append(torch.cat(gather_x).detach().cpu().numpy())
                syn_labels.append(torch.cat(gather_y).detach().cpu().numpy())

        # If this is the main process, finalize the generation process
        if self.global_rank == 0:
            logging.info("Generation Finished!")
            
            # Concatenate all collected synthetic data and labels
            syn_data = np.concatenate(syn_data)
            syn_labels = np.concatenate(syn_labels)

            if 'pe_last' in config and config['pe_last']:
                logging.info("PE selecting at last start!")
                syn_data, syn_labels, _, _ = self.pe_vote(torch.from_numpy(syn_data), syn_labels, None, self.sensitive_features.numpy(), self.sensitive_labels.numpy(), selection_ratio=config.selection_ratio, config=self.all_config, device=self.device)
                syn_data = syn_data.numpy()
                logging.info("PE selecting at last end!")
            
            np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)
            
            show_images = []
            for cls in range(self.private_num_classes):
                show_images.append(syn_data[syn_labels==cls][:8])
            show_images = np.concatenate(show_images)
            
            torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
            
            # Return the synthetic data and labels
            return syn_data, syn_labels
        else:
            return None, None