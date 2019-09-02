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
# Expect outputs and labels to have same shape (ie: torch.Size([batch:1, 224, 224])), and type long
def iou_segmentation(outputs: torch.Tensor, labels: torch.Tensor):    
    # Will be zero if Truth=0 or Prediction=0 
    intersection = (outputs & labels).float().sum((1, 2))    
    # Will be zzero if both are 0   
    union = (outputs | labels).float().sum((1, 2))          
    
    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)      
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch