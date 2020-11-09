import torch 
import torch.nn as nn
from collections import defaultdict
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
        
    def __call__(self, yoloOutput: list) -> dict:
        result = defaultdict()
        for order, layer in enumerate(yoloOutput):

            mask = layer["mask"]
            loss_x = self.pos_loss(layer["x"][0][mask], layer["x"][1][mask])
            loss_y = self.pos_loss(layer["y"][0][mask], layer["y"][1][mask])
            loss_w = self.pos_loss(layer["w"][0][mask], layer["w"][1][mask])
            loss_h = self.pos_loss(layer["h"][0][mask], layer["h"][1][mask])

            loss_conf = self.conf_loss(
                layer["conf"][0][mask], layer["conf"][1][mask])*self.lb_obj \
                + self.conf_loss(layer["conf"][0][1 - mask],
                                layer["conf"][1][1 - mask])*self.lb_noobj


            loss_class = self.conf_loss(
                layer["class"][0][mask], torch.argmax(layer["class"][1][mask], 1))

            total_loss = (loss_x + loss_y + loss_w + loss_h)*self.lb_pos + loss_conf + loss_class*self.lb_class

            result["Layer_" + str(layer)] = {}

            result["Layer_" + str(order)].update({'loss' : total_loss})
            result["Layer_" + str(order)].update({'loss_x' : loss_x})
            result["Layer_" + str(order)].update({'loss_y' : loss_y})
            result["Layer_" + str(order)].update({'loss_w' : loss_w})
            result["Layer_" + str(order)].update({'loss_h' : loss_h})
            result["Layer_" + str(order)].update({'loss_conf': loss_conf})
            result["Layer_" + str(order)].update({'loss_class': loss_class})

