import torch
import random
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset
from data.stylegan3.dataset import ImageFolderDataset

def obtain_path(dataset_name):

    if dataset_name == 'mnist':
        train_path = 'dataset/mnist/train_28.zip'
        test_path =  'dataset/mnist/test_28.zip'
        resolution = 28
        channel = 1

    elif dataset_name == 'cifar10':
        train_path = 'dataset/cifar10/train_32.zip'
        test_path =  'dataset/cifar10/test_32.zip'
        resolution = 32
        channel = 3

    elif dataset_name == 'covidx':
        train_path = 'dataset/covidx-cxr4/train_512.zip'
        test_path =  'dataset/covidx-cxr4/train_512.zip'
        resolution = 512
        channel = 3

    elif dataset_name == 'celeba_male':
        train_path = 'dataset/celeba/train_256_Male.zip'
        test_path =  'dataset/celeba/test_256_Male.zip'
        resolution = 256
        channel = 3

    elif dataset_name == 'camelyon':
        train_path = 'exp/train_96.zip'
        test_path =  'exp/test_96.zip'
        resolution = 96
        channel = 3

    elif dataset_name == 'octmnist':
        train_path = 'dataset/octmnist/train_128.zip'
        test_path =  'dataset/octmnist/test_128.zip'
        resolution = 128
        channel = 1

    else:
        print(f"Error: '{dataset_name}' is not a valid dataset name.")
        return

    return train_path, test_path, resolution, channel


def load_sensitive_data(dataset_name):  

    train_path, test_path, resolution, channel = obtain_path(dataset_name)
    
    print(train_path)  
    sensitive_train_set = ImageFolderDataset(
            train_path, resolution, channel, use_labels=True)
    sensitive_test_set = ImageFolderDataset(
            test_path, resolution, channel, use_labels=True)
    sensitive_train_set.data_name = dataset_name

    batch_size = 100
    
    sensitive_train_loader = torch.utils.data.DataLoader(dataset=sensitive_train_set, shuffle=True, drop_last=False, batch_size=batch_size)
    sensitive_test_loader = torch.utils.data.DataLoader(dataset=sensitive_test_set, shuffle=False, drop_last=False, batch_size=batch_size)

    return sensitive_train_loader, sensitive_test_loader