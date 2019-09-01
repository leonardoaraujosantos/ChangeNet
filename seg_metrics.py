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


SMOOTH = 1e-6

def iou_binary(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = (outputs > 0).squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

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


def dice(model_outputs, labels):
    model_outputs = model_outputs.clone()
    labels = labels.clone()
    
        
    smooth = 1e-4

    iflat = model_outputs.contiguous().view(-1)
    tflat = labels.contiguous().view(-1)
    intersection = torch.abs(iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

