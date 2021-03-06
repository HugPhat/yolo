import os
import random

import cv2
import numpy as np


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .data_argument import custom_aug

import matplotlib.pyplot as plt 

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
                        max_objects=10,
                        is_train=True,
                        save_draw_images = False,
                        split=None,
                        ):
        self.image_aug = custom_aug()
        self.img_size = img_size
        self.img_shape = (img_size, img_size)
        self.argument = argument
        self.max_objects = max_objects
        self.debug = debug
        self.draw = draw
        self.labels = labels
        self.path = path
        self.is_train = is_train
        self.split = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        self.denormalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.255]),
        ])
        self._file = os.path.dirname(__file__)
        self.rawData = self.InitDataset()
        self.save_draw_images = save_draw_images

    def InitDataset(self, **kwargs):
        """ Init and Read all file names

        Args:
            path (str): path to dataset

        Raises:
            NotImplementedError: [description]

        Return:
            self.rawData -> List:[[image_data, label]]
        """
       
        raise NotImplementedError()

    def GetData(self, index, **kwargs):
        """ Abstract method to get single data with yolo data

        Args:
            index ([int]): index element 

        Raises:
            NotImplementedError: Need to implement in child class
        Returns:
            image: np.ndarray
            bboxes: list
            fname: name of xml file
        """
        
        raise NotImplementedError()
    
    def content(self, idx, draw):
        '''
        input: idx of self.pairData

        output: img:np, bbox:np, name:list
        '''

        img, bbox, fname  = self.GetData(idx)
        if self.debug:
            print(f'file {fname}')
            print('-> origin: {}'.format(bbox))

        Name = [each[0] for each in bbox]
        name = [self.labels.index(each[0]) for each in bbox]
        bbox = [each[1:] for each in bbox]
        #bbox = np.asarray(bbox).astype('float32')

        if self.is_train:
            #img, bbox, name = self.image_aug(img, bbox, name)
            if random_random(0.6):
                img, bbox, name = self.image_aug(img, bbox, name)
            else:
                bbox = np.asarray(bbox)
        else:
            bbox = np.asarray(bbox)
        if self.debug:
            print(f' ---> after aug: {bbox} {type(bbox)}')

        h, w = img.shape[:2] 
        #
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else (
            (0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img.copy(), pad, 'constant', constant_values=0)
        
        padded_h, padded_w, _ = input_img.shape

        img = cv2.resize(input_img, self.img_shape)

        H, W = img.shape[:2] # new image shape

        bbox[:, 0] = (bbox[:, 0] + pad[1][0]) * 1 / (padded_w / W)
        bbox[:, 2] = (bbox[:, 2] + pad[1][0]) * 1 / (padded_w / W)
        bbox[:, 1] = (bbox[:, 1] + pad[0][0]) * 1 / (padded_h / H)
        bbox[:, 3] = (bbox[:, 3] + pad[0][0]) * 1 / (padded_h / H)

        bbox = np.asarray(bbox).astype(float)
        _bbox = bbox.copy()

        bbox[:, 0] = ((_bbox[:, 0] + _bbox[:, 2]) / 2)  * 1 / W  # xc
        bbox[:, 1] = ((_bbox[:, 1] + _bbox[:, 3]) / 2)  * 1 / H  # yc
        bbox[:, 2] = (_bbox[:, 2] - _bbox[:, 0])        * 1 / W  # w
        bbox[:, 3] = (_bbox[:, 3] - _bbox[:, 1])        * 1 / H  # h

        name = np.asarray(name).astype('float32')

        name = np.expand_dims(name, axis=0)

        label = np.concatenate((name.T, bbox), axis=1)
        if self.debug:
            print(f' ---> cvt label: {label}')
        if self.draw:
            self.drawConvertedAnnotation(img, label, fname, save=self.save_draw_images)
        return img, label, Name

    def __getitem__(self, index):

        img, labels, name = self.content(index, draw=self.draw)

        input_img = self.transform(img.copy()/255.0).float()
        labels = labels.reshape(-1, 5)
        
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]
                          ] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        if self.debug:
            print('final label ', filled_labels)
        return input_img, filled_labels

    def __len__(self):
        return len(self.rawData)

    def drawConvertedAnnotation(self, image, labels, fname, save=False):
        H, W = image.shape[:2]
        t_image = image.copy()
        for i in range(len(labels)):
            name, xc, yc, w, h = labels[i]
            x1 = int((xc - w/2)*W)
            y1 = int((yc - h/2)*H)
            x2 = int((xc + w/2)*W)
            y2 = int((yc + h/2)*H)
            print(f'pseudo {(x1, y1, x2, y2)}')
            cv2.rectangle(t_image, (x1, y1), (x2, y2), (0, 255, 255), 2, 1)
            cv2.putText(t_image, str(self.labels[int(name)]),
                        (x1+10, y1+10 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 5), 2)
        if not save:
            plt.imshow(t_image)
            plt.show()
        else:
            t_image = cv2.cvtColor(t_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self._file, "Test", "data", fname+'.jpg'))
        
