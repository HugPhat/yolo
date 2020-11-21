import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

import torch

from Yolov3.Model.yolov3 import yolov3


def create_model(num_classes, 
        default_cfg:str = None ,
        lb_noobj = 1.0,
        lb_obj=5.0,
        lb_class=2.0,
        lb_pos=1.0
    ):
    """[Create yolo model]

    Args:
        num_classes (int): [number of class when using default config]
        default_cfg (str, None): [path to custom config]. Defaults to None.

    Returns:
        [yolov3]: [model yolo]
    """
    if default_cfg:
        return yolov3(classes= num_classes, 
                      lb_noobj=lb_noobj,
                      lb_obj=lb_obj,
                      lb_class=lb_class,
                      lb_pos=lb_pos)
    else:
        return yolov3(use_custom_config=default_cfg, 
                      lb_noobj=lb_noobj,
                      lb_obj=lb_obj,
                      lb_class=lb_class,
                      lb_pos=lb_pos)
