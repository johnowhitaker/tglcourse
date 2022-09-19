# AUTOGENERATED! DO NOT EDIT! File to edit: ../61_Datasets.ipynb.

# %% auto 0
__all__ = ['to_tensor', 'mnist_transform', 'get_mnist_dl']

# %% ../61_Datasets.ipynb 3
import torch
import datasets
from .utils import *
from datasets import load_dataset
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

# %% ../61_Datasets.ipynb 14
# Specify the pre-processing
to_tensor = T.ToTensor()
def mnist_transform(example):
    example["image"] = [to_tensor(image) for image in example["image"]]
    return example

# Re-create the streaming example above
def get_mnist_dl(batch_size=32, streaming=True, split='train'):
    mnist_dataset = load_dataset('mnist', split=split, streaming=streaming)
    if streaming:
        mnist_dataset = mnist_dataset.map(mnist_transform, batch_size=batch_size, batched=True)
        mnist_dataset = mnist_dataset.with_format("torch")
    else:
        mnist_dataset = mnist_dataset.with_transform(transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size)
    return dataloader
