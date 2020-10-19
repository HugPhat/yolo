import os
import sys
import random
from copy import copy

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .data_argument import random_blur, random_brightness, random_hue,\
                 random_rotate, random_shear, random_HFlip, random_VFlip, random_scale, random_saturation

def random_random(v=0.5):
    if random.random() > 0.5:
        return True
    else:
        return False


class yoloCoreDataset(Dataset):
    def __init__(self,  path, 
                        labels, # List of labels name 
                        img_size= 416, # fixed size image
                        debug=False, 
                        argument=True, 
                        draw=False, 
                        max_objects=5
                        ):
        self.img_size = img_size
        self.img_shape = (img_size, img_size)
        self.argument = argument
        self.max_objects = max_objects
        self.debug = debug
        self.draw = draw
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        self.denormalize_image = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.255]),
        ])
        
        self.rawData = self.InitDataset(path)

    def InitDataset(self, path, **kwargs):
        '''
        * Input: 
            + path-> string: path to data
        * Ouput:
            + self.rawData -> List:
                [[image_data, label]]
        '''    
        raise NotImplementedError()

    def GetData(self, index, **kwargs):
        '''
        * Input:
            + index: index in list of data
        * Output:
            + image -> np.ndarray: image data 
            + bbox -> np.ndarray: bounding boxes shape (num, 4)

        '''
        
        raise NotImplementedError()
    
    def content(self, idx, draw):
        '''
        input: idx of self.pairData

        output: img:np, bbox:np, name:list
        '''

        img, bbox  = self.GetData(idx)
        
        if self.argument:
            if random_random(0.3):
                img, bbox = random_HFlip(img, bbox)
            if random_random(0.3):
                img, bbox = random_VFlip(img, bbox)
            if random_random(0.5):
                random_blur(img)
            if random_random(0.4):
                img, bbox = random_rotate(img, bbox)
            if random_random(0.2):
                random_saturation(img)
            if random_random()0.4:
                img, bbox = random_shear(img, bbox)
            if random_random(0.5):
                random_brightness(img)
            if random_random(0.24):
                random_hue(img)

        h, w = img.shape[:2] 
        #
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else (
            (0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img.copy(), pad, 'constant', constant_values=128)
        
        padded_h, padded_w, _ = input_img.shape

        img = cv2.resize(input_img, self.img_shape)

        H, W = img.shape[:2] # new image shape

        Name = [each[0] for each in bbox]
        bbox = [each[1:] for each in bbox]

        bbox[0][0] = (bbox[0][0] + pad[1][0]) * 1 / (padded_w / W)
        bbox[0][1] = (bbox[0][1] + pad[0][0]) * 1 / (padded_h / H)
        bbox[0][2] = (bbox[0][2] + pad[1][0]) * 1 / (padded_w / W)
        bbox[0][3] = (bbox[0][3] + pad[0][0]) * 1 / (padded_h / H)
        
        bbox = np.asarray(bbox).astype(float)
        _bbox = bbox.copy()

        bbox[:, 0] = ((_bbox[:, 0] + _bbox[:, 2]) / 2)    # xc
        bbox[:, 1] = ((_bbox[:, 1] + _bbox[:, 3]) / 2)    # yc
        bbox[:, 2] = (_bbox[:, 2] - _bbox[:, 0])          # w
        bbox[:, 3] = (_bbox[:, 3] - _bbox[:, 1])          # h

        _bbox = bbox.copy()
        bbox[:, 0] = _bbox[:, 0] * 1 / W
        bbox[:, 2] = _bbox[:, 2] * 1 / W
        bbox[:, 1] = _bbox[:, 1] * 1 / H
        bbox[:, 3] = _bbox[:, 3] * 1 / H

        name = np.asarray(Name)

        name = np.expand_dims(name, axis=0)

        label = np.concatenate((name.T, bbox), axis=1)

        return img, label, Name

    def __getitem__(self, index):

        img, labels, name = self.content(index, draw=self.draw)

        input_img = self.transform(img.copy()).float()
        labels = labels.reshape(-1, 5)

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        
        if self.debug:
            print(name)
            print(labels)

        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]
                          ] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return input_img, filled_labels

    def __len__(self):
        return len(self.rawData)

