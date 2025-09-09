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
        test_path =  'dataset/covidx-cxr4/test_512.zip'
        resolution = 512
        channel = 3

    elif dataset_name == 'celeba':
        train_path = 'dataset/celeba/train_256_Male.zip'
        test_path =  'dataset/celeba/test_256_Male.zip'
        resolution = 256
        channel = 3

    elif dataset_name == 'camelyon':
        train_path = 'dataset/covidx-cxr4/train_512.zip'
        test_path =  'dataset/covidx-cxr4/test_512.zip'
        resolution = 64
        channel = 3

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

    batch_size = 100
    
    # if config.eval.mode == "val":
    #     # split the sensitive dataset into training set and validation set
    #     if "mnist" in config.sensitive_data.name:
    #         train_size = 55000
    #     elif "cifar" in config.sensitive_data.name:
    #         train_size = 45000
    #     elif "eurosat" in config.sensitive_data.name:
    #         train_size = 21000
    #     elif "celeba" in config.sensitive_data.name:
    #         train_size = 145064
    #     elif "camelyon" in config.sensitive_data.name:
    #         train_size = 269538
    #     elif "covidx" in config.sensitive_data.name:
    #         train_size = 67863
    #     else:
    #         raise NotImplementedError

    # train_size = len(sensitive_train_set)
    #     torch.manual_seed(0)
    #     sensitive_train_set, sensitive_val_set = random_split(sensitive_train_set, [train_size, val_size])
    #     sensitive_val_loader = torch.utils.data.DataLoader(dataset=sensitive_val_set, shuffle=False, drop_last=False, batch_size=config.eval.batch_size)
    #     print("train size: {} val size: {}".format(len(sensitive_train_set), len(sensitive_val_set)))
    # else:
    #     sensitive_val_set = None
    #     sensitive_val_loader = None
    

    sensitive_train_loader = torch.utils.data.DataLoader(dataset=sensitive_train_set, shuffle=True, drop_last=False, batch_size=batch_size)
    sensitive_test_loader = torch.utils.data.DataLoader(dataset=sensitive_test_set, shuffle=False, drop_last=False, batch_size=batch_size)

    return sensitive_train_loader, sensitive_test_loader