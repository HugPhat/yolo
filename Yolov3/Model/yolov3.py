import os
from collections import defaultdict

import numpy as np

import torch
from torch import FloatTensor
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

from .headv3 import yoloHeadv3


class skip_connection(nn.Module):
    def __init__(self):
        super(skip_connection, self).__init__()



class yolov3(nn.Module):
    def __init__(self,
                 img_size=416,
                 debug=False,
                 classes = 80,
                 use_custom_config =False ,
                 config_path='config/model.cfg',
                 lb_noobj=1.0,
                 lb_obj=5.0,
                 lb_class=2.0,
                 lb_pos=1.0
                 ):
        super(yolov3, self).__init__()
        self.lb_noobj = lb_noobj
        self.lb_obj = lb_obj
        self.lb_class = lb_class
        self.lb_pos = lb_pos
        self.debug = debug
        if use_custom_config:
            self.layer_dict = self.get_config(use_custom_config)
        else:
            pwd = os.path.dirname(__file__)
            pwd = os.path.join(pwd, "yolov3.cfg")
            self.layer_dict = self.get_default_config(pwd, classes=classes)
        self.make_nn(debug=debug)
        self.loss_names = ["x", "y", "w", "h",
                           "object", "class", "recall", "precision"]
        self.seen = 0
        self.header = np.array([0, 0, 0, self.seen, 0])
        
        self.losses_log = []

    def make_conv(self, module, block, prev, it, debug=True):
        try:
            bn = int(block['batch_normalize'])
            bias = False
        except:
            bn = 0
            bias = True
        f = int(block['filters'])
        s = int(block['stride'])
        ks = int(block['size'])
        pad = int((ks-1)//2) if int(block['pad']) else 0

        conv = nn.Conv2d(prev, f, kernel_size=ks,
                         padding=pad, stride=s, bias=bias)
        module.add_module(f'CONV2D_{it}', conv)
        if bn:
            module.add_module(f'BN_{it}', nn.BatchNorm2d(f))
        act = block['activation']
        if act == 'leaky':
            module.add_module(f'LeakyReLU_{it}',
                              nn.LeakyReLU(0.1, inplace=True))
        elif act == 'relu':
            module.add_module(f'ReLU_{it}', nn.ReLU(inplace=True))
        if debug:
            print(
                f'Making conv_{it}, with f={f}, s={s}, ks={ks}, p={pad}, out: {f}')
        prev = f
        return f

    def make_upsample(self, module, block, prev, it, debug=True):
        s = int(block['stride'])
        if debug:
            print(f'Making upsample_{it} with scale_factor {s}')
        module.add_module(f'UPSAMPLE_{it}', nn.Upsample(
            scale_factor=s, mode='nearest'))
        return prev

    def make_route(self, module, block, prev, it, debug=True):
        route = block['layers'].split(',')
        start = int(route[0])
        if len(route) > 1:
            end = int(route[1])
            start = (it - start) if start > 0 else start
            end = (end - it) if end > 0 else end
            filters = self.output_filter[start] + self.output_filter[end]
        else:
            start = (it - start) if start > 0 else start
            filters = self.output_filter[start]
        module.add_module(f'ROUTE_{it}', skip_connection())

        if debug:
            print(f'at it: {it} ROUTE {route}, filters = {filters}')
        return filters

    def make_shortcut(self, module, block, prev, it, debug=True):
        from_layer = int(block['from'])
        activation = block['activation']
        if debug:
            print(
                f'make shortcut {it} from layer {from_layer} with activation {activation}')
        filters = prev  # self.output_filter[it + from_layer]
        #self.output_filter[it] = filters
        if activation == 'linear':
            l = skip_connection()
        module.add_module(f'SHORTCUT_{it}', l)
        return filters

    def make_maxpool(self, module, layer, prev, it, debug):
        stride = int(layer['stride'])
        size = int(layer['size'])
        pad = int((size - 1) // 2)
        if debug:
            print(f'make MAXPOOL_{it} stride = {stride}, size = {size}')
        if size == 2 and stride == 1:  # yolov3-tiny
            module.add_module(f'ZeroPad2d_{it}', nn.ZeroPad2d((0, 1, 0, 1)))
            #module.add_module(f'MAXPOOL_{it}', nn.MaxPool2d(stride=stride, kernel_size=size, padding=pad))
        #else:
        module.add_module(f'MAXPOOL_{it}', nn.MaxPool2d(
            stride=stride, kernel_size=size, padding=pad))

        return prev

    def make_yolo_layer(self, module, layer, prev, it, debug):
        mask = layer['mask'].split(',')
        anchor = layer['anchors'].split(',')

        masks = [int(each) for each in mask]
        anchors = [[int(anchor[i]), int(anchor[i+1])]
                   for i in range(0, len(anchor), 2)]
        anchors = [anchors[i] for i in masks]

        num_classes = int(layer['classes'])
        num_anchors = int(layer['num'])
        yolo = yoloHeadv3(anchors, num_classes, img_size=int(
            self.hyperparameters['width']),
            lb_noobj=self.lb_noobj, lb_obj=self.lb_obj, lb_class=self.lb_class, lb_pos=self.lb_pos)
        module.add_module(f"yolo_detection_{it}", yolo)
        if debug:
            print(f'make yolo layer with mask :{mask}, classes: {num_classes}')
        return prev

    def make_nn(self, debug=True):
        self.hyperparameters = self.layer_dict[0]  # type:'net'
        self.output_filter = []
        self.yolov3 = nn.ModuleList()
        prev_filter = int(self.hyperparameters['channels'])
        self.img_size = int(self.hyperparameters['width'])
        self.output_filter.append(prev_filter)
        for it, layer in enumerate(self.layer_dict[1:]):  # first is net_info
            #print(layer)
            _type = layer['type']
            module = nn.Sequential()
            if _type == 'convolutional':
                filters = self.make_conv(module, layer, prev_filter, it, debug)
            elif _type == 'maxpool':
                filters = self.make_maxpool(
                    module, layer, prev_filter, it, debug)
            elif _type == 'upsample':
                filters = self.make_upsample(
                    module, layer, prev_filter, it, debug)
            elif _type == 'route':
                filters = self.make_route(
                    module, layer, prev_filter, it, debug)
            elif _type == 'shortcut':
                filters = self.make_shortcut(
                    module, layer, prev_filter, it, debug)
            elif _type == 'yolo':
                filters = self.make_yolo_layer(
                    module, layer, prev_filter, it, debug)
            else:
                raise Exception('UNKNOWN FORMAT')
            self.output_filter.append(filters)
            self.yolov3.append(module)
            prev_filter = filters
        if debug:
            print(self.yolov3)

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        layer_output = []
        if is_training:
            self.losses_log = []
        count = 0
        for i, (layer_dict, module) in enumerate(zip(self.layer_dict[1:], self.yolov3)):
            t = layer_dict['type']
            if self.debug:
                print(
                    f'we are at [{i}] th layer type[ {t}] with info {layer_dict}')
            if layer_dict['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif layer_dict['type'] == 'route':
                layer = [int(item) for item in layer_dict['layers'].split(',')]
                x = torch.cat([layer_output[l] for l in layer], 1)
                #    x = torch.cat([layer_output[-1], layer_output[layer[0]]])
            elif layer_dict['type'] == 'shortcut':
                _from = int(layer_dict['from'])
                x = layer_output[-1] + layer_output[_from]
            elif layer_dict['type'] == 'yolo':
                if is_training:
                    #x, *losses = module[0](x, targets)
                    x, losses = module[0](x, targets)
                    self.losses_log.append(losses)

                else:
                    count += 1
                    if count:
                        x = module(x)
                output.append(x)

            layer_output.append(x)

        return sum(output) if is_training else torch.cat(output, 1)

    def get_config(self, path):
        with open(path, 'r') as cfg:
            lines = cfg.read().split('\n')
            if self.debug:
                print(type(lines))
            lines = [x for x in lines if len(x) > 0]
            lines = [x for x in lines if x[0] != '#']
            lines = [x.rstrip().lstrip() for x in lines]
        block = {}
        blocks = []
        for line in lines:
            if line[0] == "[":
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
        return blocks
    
    def get_default_config(self, path, classes):
        with open(path, 'r') as cfg:
            lines = cfg.read().split('\n')
            if self.debug:
                print(type(lines))
            lines = [x for x in lines if len(x) > 0]
            lines = [x for x in lines if x[0] != '#']
            lines = [x.rstrip().lstrip() for x in lines]
        block = {}
        blocks = []
        for line in lines:
            if line[0] == "[":
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                value = value.lstrip()
                if value == "$filters":
                    value = str((classes + 5) * 3)
                elif value == "$classes":
                    value = str(classes)
                    
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
        
        return blocks

    def load_weight(self, weights_path='config/yolov3_tiny.weights'):

        with open(weights_path, 'rb') as fw:
            header = np.fromfile(fw, dtype=np.int32, count=5)
            self.header = header
            self.seen = header[3]
            weights = np.fromfile(fw, dtype=np.float32)
        ptr = 0
        for i, (layer_dict, module) in enumerate(zip(self.layer_dict[1:], self.yolov3)):

            if layer_dict['type'] == 'convolutional':
                conv_layer = module[0]
                #try:
                try:
                    #if layer_dict["batch_normalize"]:
                    xx = layer_dict["batch_normalize"]
                except:
                    xx = 0
                if xx:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                #except:
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
