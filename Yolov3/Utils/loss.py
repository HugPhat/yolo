# custom loss

import torch 
import torch.nn as nn 

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, target):
        pass