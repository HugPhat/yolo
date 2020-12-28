import torch 
import torch.optim as optim 
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

from Yolov3.Model.yolov3 import yolov3 as yolo 
from Yolov3.Utils.train_module import train_module
from simple_data import simple_data as dataSet 

model = yolo(classes=2, use_custom_config=False, lb_obj=1, lb_class=1, lb_noobj=0)

path_to_data = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\simple_obj_detection\cat_dog'

trainLoader = DataLoader(dataSet(path=path_to_data, labels=['cat', 'dog'], max_objects=3, split=0.02,
                                 debug=False, draw=False, argument=True, is_train=True),
                         batch_size=8,
                         shuffle=True,
                         num_workers=0,
                         drop_last=False
                         )

valLoader = DataLoader(dataSet(path=path_to_data, labels=['cat', 'dog'], max_objects=3, split=0.9,
                               debug=False, draw=False, argument=False, is_train=False),
                       batch_size=8,
                       shuffle=True,
                       num_workers=0,
                       drop_last=False
                       )

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.92, weight_decay=5e-4)

train_module(
    model=model,
    trainLoader=trainLoader,
    valLoader=valLoader,
    optimizer_name='sgd',
    optimizer=optimizer,
    lr_scheduler=None,
    warmup_scheduler=None,
    Epochs=10,
    use_cuda= False,
    writer=None,
    path=None,
    lr_rate=0.001,
    wd=5e-5,
    momen=0.92,
    start_epoch=1
)
