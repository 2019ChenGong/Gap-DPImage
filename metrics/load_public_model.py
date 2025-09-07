import torch
import random
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset
from data.stylegan3.dataset import ImageFolderDataset


def load_public_model(public_model):  
    
    return