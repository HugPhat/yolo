import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


def Train(
    model,  # yolov3
    trainLoader: DataLoader, # train: DataLoader
    valLoader : DataLoader, # val: DataLoader
    optimizer : DataLoader, # optimizer 
    lr_schedule,
    warmup_schedule,
    loss_function, # Loss function
    
    num_epoch : int,
    
)
