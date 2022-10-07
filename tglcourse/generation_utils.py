# AUTOGENERATED! DO NOT EDIT! File to edit: ../62_Generators_and_Losses.ipynb.

# %% auto 0
__all__ = ['MSELossToTarget', 'ImStackGenerator', 'calc_vgg_features', 'ContentLossToTarget']

# %% ../62_Generators_and_Losses.ipynb 7
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from matplotlib import pyplot as plt
from imstack.core import ImStack
from siren_pytorch import Sine, Siren, SirenNet, SirenWrapper
import open_clip
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
import torchvision.models as models

# %% ../62_Generators_and_Losses.ipynb 16
class MSELossToTarget(nn.Module):
  """ MSE between input and target, resizing if needed"""
  def __init__(self, target, size=256):
    super(MSELossToTarget, self).__init__()
    self.resize = T.Resize(size)
    self.target = self.resize(target) # resize target image to size

  def forward(self, input):
    input = self.resize(input) # set size (assumes square images)
    squared_error = (self.target - input)**2
    return squared_error.mean() # MSE

# %% ../62_Generators_and_Losses.ipynb 28
class ImStackGenerator(nn.Module):
  """An imstack to represent the image"""
  def __init__(self, size=256, n_layers=4, base_size=16, 
               layer_decay=0.7, init_image=None, scale=2):
    super(ImStackGenerator, self).__init__()
    self.imstack = ImStack(n_layers=n_layers, out_size=size, 
                           base_size=base_size, init_image=init_image, 
                           decay=layer_decay, scale=scale)
  
  def parameters(self): # How to access the learnable parameters
    return self.imstack.layers

  def forward(self):
    return self.imstack()

# %% ../62_Generators_and_Losses.ipynb 36
# Extracting features from an image using this pretrained model:
def calc_vgg_features(imgs, use_layers=[1, 6, 11, 18, 25]):
  mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None].to(device)
  std = torch.tensor([0.229, 0.224, 0.225])[:,None,None].to(device)
  x = (imgs-mean) / std
  b, c, h, w = x.shape
  features = [x.reshape(b, c, h*w)] # This reshape is for convenience later
  for i, layer in enumerate(vgg16[:max(use_layers)+1]):
    x = layer(x)
    if i in use_layers:
      b, c, h, w = x.shape
      features.append(x.reshape(b, c, h*w))
  return features

# %% ../62_Generators_and_Losses.ipynb 41
class ContentLossToTarget(nn.Module):
  """ Perceptual loss between input and target, resizing if needed, based on vgg16"""
  def __init__(self, target, size=128, content_layers = [14, 19]):
    super(ContentLossToTarget, self).__init__()
    self.resize = T.Resize(size)
    self.target = self.resize(target) # resize target image to size
    self.content_layers = content_layers
    with torch.no_grad():
      self.target_features = calc_vgg_features(self.target, use_layers = self.content_layers)

  def forward(self, input):
    input = self.resize(input) # set size (assumes square images)
    input_features = calc_vgg_features(input, use_layers = self.content_layers)
    l = 0
    # Run through all features and take l1 loss (mean error) between them
    for im_features, target_features in zip(input_features, self.target_features):
      l += nn.L1Loss()(im_features, target_features)
    return l/len(input_features)
