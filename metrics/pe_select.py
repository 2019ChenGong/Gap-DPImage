from metrics.dp_metrics import DPMetric
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

import os
import shutil
from models.DP_Diffusion.model.ncsnpp import NCSNpp
from models.DP_Diffusion.utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid, compute_fid_with_images
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor

from models.dp_merf import DP_MERF as Freq_Model
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset

from models.PE.pe.dp_counter import dp_nn_histogram
from models.PE.apis.improved_diffusion.gaussian_diffusion import create_gaussian_diffusion

class PE_Select(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)

        self.n_dim = 12
        self.vec_size = self.max_images
        self.sensitive_dataset = sensitive_dataset
        self.data_name = sensitive_dataset.dataset.data_name

        self.inception_model = InceptionFeatureExtractor()
        self.inception_model.model = self.inception_model.model.to(self.device)

        for batch, _ in self.sensitive_dataset:
            self.image_height = batch.shape[2]
            self.image_width = batch.shape[3]
            break
    
    def cal_metric(self, args):
        time = self.get_time()
        gen_num = 50

        sensitive_features = []
        sensitive_labels = []
        sensitive_dataloader_inc = DataLoader(self.sensitive_dataset.dataset, batch_size=100)
        for x, y in sensitive_dataloader_inc:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            features_batch = self.inception_model.get_feature_batch(x.to(self.device))
            sensitive_features.append(features_batch.detach().cpu())
            sensitive_labels.append(y)
        sensitive_features = torch.cat(sensitive_features).numpy()
        sensitive_labels = torch.cat(sensitive_labels).numpy()

        public_features = []
        public_labels = []

        self.public_model.model_id = "Manojb/stable-diffusion-2-1-base"
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-0"
        generation_dataloader1 = self._image_generation(save_dir, max_images=gen_num)
        print(len(generation_dataloader1.dataset))

        self.public_model.model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-1"
        generation_dataloader2 = self._image_generation(save_dir, max_images=gen_num)

        self.public_model.model_id = "CompVis/stable-diffusion-v1-4"
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-2"
        generation_dataloader3 = self._image_generation(save_dir, max_images=gen_num)

        for x, _ in generation_dataloader1:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            features_batch = self.inception_model.get_feature_batch(x.to(self.device))
            public_features.append(features_batch.detach().cpu())
            public_labels += [0]*len(features_batch)
        for x, _ in generation_dataloader2:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            features_batch = self.inception_model.get_feature_batch(x.to(self.device))
            public_features.append(features_batch.detach().cpu())
            public_labels += [1]*len(features_batch)
        for x, _ in generation_dataloader3:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            features_batch = self.inception_model.get_feature_batch(x.to(self.device))
            public_features.append(features_batch.detach().cpu())
            public_labels += [2]*len(features_batch)
        public_features = torch.cat(public_features).numpy()
        public_labels = torch.tensor(public_labels).long()

        voting_results = self.pe_vote(public_features, public_labels, 3, sensitive_features, sensitive_labels, None)
        print(voting_results)
    
    def pe_vote(self, features_to_selected, image_categories, model_num, sensitive_features, sensitive_labels, config, sigma=5, num_nearest_neighbor=1, nn_mode='L2', count_threshold=4.0, selection_ratio=0.1, device=None, sampler=None):


        sub_count, sub_clean_count = dp_nn_histogram(
            public_features=features_to_selected,
            private_features=sensitive_features,
            noise_multiplier=sigma,
            num_nearest_neighbor=num_nearest_neighbor,
            mode=nn_mode,
            threshold=count_threshold,
            device=0,
            verbose=False,
        )

        torch.cuda.empty_cache()

        voting_results = {np.sum(sub_count[image_categories==cls]) for cls in range(model_num)}

        return voting_results