# Pytorch stuff
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import torch.utils.data as utils
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

# This version uses hard threshold so it can't be used on the loss function but is precise 
# to verify the real IoU
# Reference: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def iou(model_outputs, labels):
    smooth = 1e-4
    bin_model_outputs = torch.zeros_like(model_outputs, requires_grad=False).type(torch.LongTensor)
    
    # Binarize label
    bin_labels = (labels > 0).type(torch.LongTensor)
    
    # Binarize output
    # Convert all channels from model_outputs to binary
    list_max = [torch.max(model_outputs[:,ch,:,:]) for ch in range(model_outputs.shape[1])]
    list_min = [torch.min(model_outputs[:,ch,:,:]) for ch in range(model_outputs.shape[1])]
    list_threshold = [(list_max[ch] - list_min[ch]) / 2.0 for ch in range(model_outputs.shape[1])]
    for ch in range(model_outputs.shape[1]):
        bin_model_outputs[:,ch,:,:] = model_outputs[:,ch,:,:] > list_threshold[ch]
    
    intersection = (bin_model_outputs & bin_labels).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (bin_model_outputs | bin_labels).float().sum((2, 3))         # Will be zzero if both are 0
    # Calculate Intersect over Union (Jaccard Index)
    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    return iou.mean()


# This version uses hard threshold so it can't be used on the loss function but is precise 
# to verify the real dice coefficient
def dice(model_outputs, labels):
    #xt = torch.FloatTensor(x,requires_grad=True)
    bin_model_outputs = torch.zeros_like(model_outputs, requires_grad=True).type(torch.FloatTensor)
    bin_labels = (labels > 0).type(torch.FloatTensor)
    
    # Convert all channels from model_outputs to binary
    list_max = [torch.max(model_outputs[:,ch,:,:]) for ch in range(model_outputs.shape[1])]
    list_min = [torch.min(model_outputs[:,ch,:,:]) for ch in range(model_outputs.shape[1])]
    list_threshold = [(list_max[ch] - list_min[ch]) / 2.0 for ch in range(model_outputs.shape[1])]
    for ch in range(model_outputs.shape[1]):
        bin_model_outputs[:,ch,:,:] = model_outputs[:,ch,:,:] > list_threshold[ch]
        
    smooth = 1e-4

    iflat = bin_model_outputs.view(-1)
    tflat = bin_labels.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))