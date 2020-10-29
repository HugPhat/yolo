import torch 
import torch.nn as nn

class yolov3Loss:
    
    def __init__(self, 
                 pos_loss = None,
                 class_loss = None,
                 conf_loss = None,
                 lb_noobj=1.0,
                 lb_obj=1.0,
                 lb_class=1.0,
                 lb_pos=1.0,
                 ):
        self.lb_noobj  = lb_noobj
        self.lb_obj    = lb_obj  
        self.lb_class  = lb_class
        self.lb_pos    = lb_pos  
        
        self.pos_loss   = nn.MSELoss() if pos_loss is None else pos_loss
        self.class_loss = nn.BCELoss() if class_loss is None else pos_loss
        self.conf_loss  = nn.CrossEntropyLoss() if conf_loss is None else pos_loss
        
    def __call__(self, yoloOuput: dict) -> dict:
        result = {}
        loss_x = self.pos_loss(yoloOuput["x"], yoloOuput["tx"])