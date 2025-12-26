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
from metrics.load_public_model import load_public_model

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
        def get_top_mask_inclusive(arr):
            arr = np.asarray(arr)
            n = len(arr)
            if n == 0:
                return np.array([], dtype=bool)
            
            k = max(1, int(np.ceil(n * selection_ratio)))
            threshold = np.partition(arr, -k)[-k]
            return arr >= threshold

        voting_results = {cls: np.sum(sub_count[image_categories==cls]) for cls in range(model_num)}
        voting_results_detail = {cls: get_top_mask_inclusive(sub_count[image_categories==cls]) for cls in range(model_num)}

        return voting_results, voting_results_detail
    
    def cal_metric(self, args):

        print("🚀 Starting PE-Select calculation...")

        time = self.get_time()
        log_dir = 'exp/results.txt'
        gen_num = 5000
        var_time = 4

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

        self.public_model.model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-0"
        generation_dataloader1 = self._image_generation(save_dir, max_images=gen_num)

        self.public_model.model_id = "Manojb/stable-diffusion-2-1-base"
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-1"
        generation_dataloader2 = self._image_generation(save_dir, max_images=gen_num)

        self.public_model.model_id = "CompVis/stable-diffusion-v1-4"
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-2"
        generation_dataloader3 = self._image_generation(save_dir, max_images=gen_num)

        self.public_model.model_id = "Manojb/stable-diffusion-2-base"
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-3"
        generation_dataloader4 = self._image_generation(save_dir, max_images=gen_num)

        self.public_model = load_public_model("dpimagebench-ldm")
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}-4"
        generation_dataloader5 = self._image_generation(save_dir, max_images=gen_num)

        public_features, public_labels = self._prepare_public_features([generation_dataloader1, generation_dataloader2, generation_dataloader3, generation_dataloader4, generation_dataloader5])

        model_names = [
            "stable-diffusion-v1-5",
            "stable-diffusion-2-1-base",
            "stable-diffusion-v1-4",
            "stable-diffusion-2-base",
            "dpimagebench-ldm"
        ]

        voting_results, voting_results_detail = self.pe_vote(public_features, public_labels, 5, sensitive_features, sensitive_labels, None)

        # Print detailed voting results for round 0
        print(f"\n{'='*80}")
        print(f"Voting Results - Round 0 (Initial Generation)")
        print(f"{'='*80}")
        for cls_id in sorted(voting_results.keys()):
            vote_count = voting_results[cls_id]
            print(f"  Model {cls_id} ({model_names[cls_id]:30s}): {vote_count:6.0f} votes")
        total_votes = sum(voting_results.values())

        with open(log_dir, 'a') as f:
            f.write(f"Round 0: {str(voting_results)}\n")

        # print(voting_results_detail)
        generation_loader_list = [generation_dataloader1, generation_dataloader2, generation_dataloader3, generation_dataloader4, generation_dataloader5]

        for var_step in range(var_time):

            generation_loader_list = self._var_one_step_multi_loader(generation_loader_list, voting_results_detail, save_dir=f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}", max_images=gen_num)

            public_features, public_labels = self._prepare_public_features(generation_loader_list)

            voting_results, voting_results_detail = self.pe_vote(public_features, public_labels, 5, sensitive_features, sensitive_labels, None)

            # Print detailed voting results for each variation round
            print(f"\n{'='*80}")
            print(f"Voting Results - Round {var_step + 1} (After Variation)")
            print(f"{'='*80}")
            for cls_id in sorted(voting_results.keys()):
                vote_count = voting_results[cls_id]
                print(f"  Model {cls_id} ({model_names[cls_id]:30s}): {vote_count:6.0f} votes")
            total_votes = sum(voting_results.values())
            print(f"{'-'*80}")
            print(f"  Total votes: {total_votes:6.0f}")
            print(f"{'='*80}\n")

            with open(log_dir, 'a') as f:
                f.write(f"Round {var_step + 1}: {str(voting_results)}\n")
            # print(voting_results_detail)
    
    def _prepare_public_features(self, data_loader_list):
        public_features = []
        public_labels = []

        for idx, data_loader in enumerate(data_loader_list):
            public_features.append(self._get_features_from_loader(data_loader))
            public_labels += [idx]*len(public_features[-1])
        
        public_features = torch.cat(public_features).numpy()
        public_labels = torch.tensor(public_labels).long()

        return public_features, public_labels

    def _var_one_step_multi_loader(self, data_loader_list, voting_results_list, save_dir='exp/tmp', max_images=None):
        new_data_loader_list = []
        for idx, data_loader in enumerate(data_loader_list):
            new_save_dir = save_dir + '-{}'.format(idx)
            generation_dataloader = self._var_one_step(data_loader, voting_results_list[idx], save_dir=new_save_dir, max_images=max_images)
            new_data_loader_list.append(generation_dataloader)

        return new_data_loader_list

    def _get_features_from_loader(self, data_loader):
        public_features = []
        for x, _ in data_loader:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            features_batch = self.inception_model.get_feature_batch(x.to(self.device))
            if features_batch.dim() == 1:
                features_batch = features_batch.unsqueeze(0)
            public_features.append(features_batch.detach().cpu())
        
        public_features = torch.cat(public_features)
        return public_features

    def _var_one_step(self, data_loader, voting_results, save_dir='exp/tmp', max_images=None):
        images = []
        for x, _ in data_loader:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            images.append(x)
        images = torch.cat(images)
        top_images = images[torch.from_numpy(voting_results).bool()]
        top_dataloader = DataLoader(TensorDataset(top_images, top_images), batch_size=10, shuffle=False, drop_last=False)

        _, generation_dataloader1 = self._image_variation(top_dataloader, save_dir, max_images=max_images)
        return generation_dataloader1