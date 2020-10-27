import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

import torch

from Yolov3.Model.yolov3 import yolov3

cfg = config_path= os.path.join(PATH, "config", "yolov3.cfg")
test_model = yolov3(classes=20)

dummy = torch.randn(1, 3, 416, 416)

y = test_model(dummy)

