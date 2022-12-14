# AUTOGENERATED! DO NOT EDIT! File to edit: ../61_Datasets.ipynb.

# %% auto 0
__all__ = ['tfm', 'mnist_transform', 'get_mnist_dl', 'imagewoof_transform', 'get_imagewoof_dl', 'cifar10_transform',
           'get_cifar10_dl']

# %% ../61_Datasets.ipynb 4
import torch
import datasets
from .utils import *
from datasets import load_dataset
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

# %% ../61_Datasets.ipynb 6
def mnist_transform(example):
    example["image"] = [T.ToTensor()(image) for image in example["image"]]
    return example

# Re-create the streaming example above
def get_mnist_dl(batch_size=32, split='train'):
    mnist_dataset = load_dataset('mnist', split=split)
    mnist_dataset = mnist_dataset.with_transform(mnist_transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size)
    return dataloader

# %% ../61_Datasets.ipynb 7
tfm = T.Compose([T.ToTensor(), T.Resize(320), T.CenterCrop(320)])
def imagewoof_transform(example):
    example["image"] = [tfm(image.convert('RGB')) for image in example["image"]]
    return example
def get_imagewoof_dl(batch_size=32):
    dataset = load_dataset('johnowhitaker/imagewoof2-320', split='train').shuffle(seed=42)
    dataset = dataset.with_transform(imagewoof_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# %% ../61_Datasets.ipynb 8
tfm = T.Compose([T.ToTensor(), T.Resize(32), T.CenterCrop(32)])
def cifar10_transform(example):
    example["image"] = [tfm(image.convert('RGB')) for image in example["image"]]
    return example
def get_cifar10_dl(batch_size=32, split='train'):
    dataset = load_dataset('cifar10', split=split).shuffle(seed=42).rename_column("img", "image")
    dataset = dataset.with_transform(cifar10_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
