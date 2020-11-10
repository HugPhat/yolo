import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

import torch

from Yolov3.Model.yolov3 import yolov3


def create_model(num_classes ):
    return yolov3(classes= num_classes).load_pretrained_by_num_class()

(create_model(20))