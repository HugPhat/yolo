from os.path import split
import sys
import os
import shutil
from datetime import datetime

File_Path = os.getcwd()
sys.path.insert(0, os.path.join(os.getcwd(), '../..'))
#sys.path.insert(1, File_Path)

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from Yolov3.Utils.train import train

from model import create_model
import data

import pytorch_warmup as warmup
from warmup_scheduler import GradualWarmupScheduler

# logger callback:
def create_writer(log_saving_path):
  writer = SummaryWriter(log_dir=log_saving_path)
  print(f'log file is saved at {log_saving_path}')
  #logfile = open(os.path.join(log_dir, 'log.txt'), 'w')
  return writer

if __name__ == "__main__":
    #######################################################################################
    import argparse
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    # freeze
    optional.add_argument('--fr', action='store_false',
                        default=True,
                        dest='freeze_backbone',
                        help='freeze pretrained backbone (True)')
    # use pretrained 
    optional.add_argument('--pre', action='store_true',
                        default=False, 
                        dest='use_pretrained',
                        help='use pretrained (False)')
    # continue trainig
    optional.add_argument('--con', action='store_false',
                        default=True, 
                        dest='to_continue',
                        help='not continue training ')
    # custom config
    optional.add_argument('--cfg', action='store',
                        default='default', type=str,
                        dest='cfg',
                        help='use custom config, if use, pass the path of custom cfg file, default is (./config/yolov3.cfg) ')
    # number of class
    optional.add_argument('--ncl', action='store',
                        default=21, type=int,
                        dest='num_class',
                        help='number of annot classes (21)')
    # number of class
    optional.add_argument('--sch', action='store_true',
                          default=False, 
                          dest='use_scheduler',
                          help='set it to turn on using scheduler (False)')
    # @@ path to voc data
    required.add_argument('--data', action='store',
                        default=None,
                        dest='data',
                        required=True,
                        help='path to voc data folder')
    # @@ split ratio of voc data
    optional.add_argument('--split', action='store',
                          default=None, type=float,
                          dest='split',
                          required=False,
                          help='split ratio [0., 1.] of voc dataset (None) if not None')
    # batch size (8)                        
    optional.add_argument('--bs', action='store',
                        default=8, type=int,
                        dest='batch_size',
                        help='number of batch size (8)')
    # number of workers (0)                        
    optional.add_argument('--nw', action='store',
                        default=0, type=int,
                        dest='num_worker',
                        help='number of worker (0)')
    # optim type                        
    optional.add_argument('--op', action='store',
                        default="sgd", type=str,
                        choices=['sgd', 'adam'],
                        dest='optim',
                        help='type of optimizer: sgd/adam (sgd)')
    # momentum for sgd
    optional.add_argument('--mo', action='store',
                        default=0.91, type=float,
                        dest='momentum',
                        help='Momentum for sgd (0.91)')
    # learning rate
    optional.add_argument('--lr', action='store',
                        default=0.01, type=float,
                        dest='lr',
                        help='learning rate (0.01)')
    # weight decay                        
    optional.add_argument('--wd', action='store',
                        default=1e-4, type=float,
                        dest='wd',
                        help='weight decay (1e-4)')
    # epoch
    optional.add_argument('--ep', action='store',
                        default=20, type=int,
                        dest='epoch',
                        help='number of epoch (20)')
    # use cuda
    optional.add_argument('--cpu', action='store_true',
                        default=False, 
                        dest='use_cpu',
                        help='use cpu or not (False)')
    # log path
    optional.add_argument('--log', action='store',
                        default="checkpoint", type=str,
                        dest='log_path',
                        help='path to save chkpoint and log (./checkpoint)')
    # lambda Objectness
    optional.add_argument('--lo', action='store',
                          default=2.0, type=float,
                          dest='lb_obj',
                          help='lambda objectness lossfunciton (2.0)')
    # lambda NoObj
    optional.add_argument('--lno', action='store',
                          default=0.5, type=float,
                          dest='lb_noobj',
                          help='lambda objectless lossfunciton (0.5)')
    # lambda position
    optional.add_argument('--lpo', action='store',
                          default=1.0, type=float,
                          dest='lb_pos',
                          help='lambda position lossfunciton (1.)')
    # lambda class
    optional.add_argument('--lcl', action='store',
                          default=1.0, type=float,
                          dest='lb_clss',
                          help='lambda class lossfunciton (1.)')
    args = parser.parse_args()
    print('Initilizing..')
    ###### handle args #########
    

    #path_2_root = r"E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012"
    path_2_root = args.data

    

    labels = data.readTxt(os.path.join(File_Path, 'config', 'class.names'))
    labels.insert(0, 0)# plus 0th: background
    trainLoader = DataLoader(data.VOC_data(path=path_2_root, labels=labels, max_objects=18, split=args.split,
                                        debug=False, draw=False, argument=False, is_train=True),
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=args.num_worker,
                             drop_last=False
                            )
    # dont use img augment for val
    valLoader = DataLoader(data.VOC_data(path=path_2_root, labels=labels, max_objects=18, split=args.split,
                                           debug=False, draw=False, argument=False, is_train=False),
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_worker,
                             drop_last=False
                             )
    
    num_steps = len(trainLoader) * args.epoch
    lr_scheduler = None
    warmup_scheduler = None

    print('Succesfully load dataset')
    if not args.cfg:  
        print(f'Succesfully load model with default config')                           
        yolo = create_model(num_classes=args.num_class,
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos
                            )
    elif args.cfg == 'default':
        f = os.path.join(File_Path, 'config', 'yolov3.cfg')
        print(f"Succesfully load model with custom config at '{f}' ")                           
        yolo = create_model(None, default_cfg=f,
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos
                            )
    else:
        print(f"Succesfully load model with custom config at '{args.cfg}' ")                           
        yolo = create_model(None, default_cfg=args.cfg, 
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos
                            )
        
        
    if args.log_path == "checkpoint":
        print('-->', yolo.model_name)
        log_folder = os.path.join(File_Path, args.log_path, yolo.model_name)
    else:
        log_folder = args.log_path

    log_file = os.path.join(log_folder, 'log')
    
    list_model = os.listdir(log_folder) if os.path.exists(log_folder) else []
    # in log folder have current model and want to resume training
    if "current_checkpoint.pth" in list_model and args.to_continue:
        checkpoint = torch.load(os.path.join(
            log_folder, "current_checkpoint.pth"), map_location=torch.device('cpu'))
        yolo.load_state_dict(checkpoint['state_dict'])
        optim_name = checkpoint["optimizer_name"]
        lr_rate = checkpoint['lr']
        wd = checkpoint['wd']
        momen = checkpoint['m']
        start_epoch = checkpoint['epoch']
        if optim_name == 'sgd':
            optimizer = optim.SGD(yolo.parameters(), 
                        lr=lr_rate,
                        weight_decay=wd, 
                        momentum=momen
                        )
            print(
                f"Use optimizer SGD with lr {lr_rate} wdecay {wd} momentum {momen}")
        else:
            optimizer = optim.Adam(yolo.parameters(),
                                lr=lr_rate,
                                weight_decay=wd,
                                )
            print(
                f"Use optimizer Adam with lr {lr_rate} wdecay {wd}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        if not args.use_cpu:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        sche_sd = checkpoint['sche']
        if sche_sd and args.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_steps)
            lr_scheduler.load_state_dict(sche_sd)
            warmup_scheduler = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=4, after_scheduler=lr_scheduler)
            #if not args.use_cpu:
            #    for state in lr_scheduler.state.values():
            #        for k, v in state.items():
            #            if isinstance(v, torch.Tensor):
            #                state[k] = v.cuda()

        print(f"Resume Trainig at Epoch {checkpoint['epoch']} ")
        writer = create_writer(log_file)
        
    else:
        print("Loading new model")
        if args.use_pretrained:
            print("Loading pretrained weight")
            yolo.load_pretrained_by_num_class()
        if args.use_pretrained  and args.freeze_backbone:
            grads_layer = ("81", "93", "105")
            for name, child in yolo.yolov3.named_children():
                if not name in grads_layer:
                    for each in child.parameters():
                        each.requires_grad = False
            print(f'Freezed all layers excludings {grads_layer}')
        if os.path.exists(log_file):
            old = datetime.now().strftime("%M%H_%d%m%Y") + "_old"
            shutil.copytree(log_file, os.path.join(log_folder, old), False, None)# copy to old folder
            shutil.rmtree(log_file, ignore_errors=True)# remove log
        lr_rate = args.lr
        wd = args.wd
        momen = args.momentum
        start_epoch = 1
        if args.optim == 'sgd':
            print(f"Use optimizer SGD with lr {lr_rate} wdecay {wd} momentum {momen}")
            optimizer = optim.SGD(yolo.parameters(), lr=lr_rate, weight_decay=wd, momentum=momen)
        elif args.optim == 'adam':
            print(f"Use optimizer Adam with lr {lr_rate} wdecay {wd} ")
            optimizer = optim.Adam(yolo.parameters(), lr=lr_rate, weight_decay=wd)
        else:
            raise f"optimizer {args.optim} is not supported"
        
        writer = create_writer(log_file)# create new log
    #set up lr scheduler and warmup
    #num_steps = len(trainLoader) * args.epoch # move to line 174
    if not lr_scheduler and args.use_scheduler:
        print('Set up scheduler and warmup')
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)
        warmup_scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=4, after_scheduler=lr_scheduler)

    print('Start training model by GPU') if not args.use_cpu else print(
        'Start training model by CPU')
    train(
        model=yolo,
        trainLoader=trainLoader,
        valLoader=valLoader,
        optimizer_name=args.optim,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
        Epochs=args.epoch,
        use_cuda= not args.use_cpu,
        writer=writer,
        path=log_folder,
        lr_rate=lr_rate,
        wd=wd,
        momen=momen,
        start_epoch=start_epoch
    )
