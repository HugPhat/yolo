import sys
import os
import shutil
from datetime import datetime

pwd_path = os.getcwd()
sys.path.insert(0, os.path.join(os.getcwd(), '../..'))
#sys.path.insert(1, pwd_path)

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from Yolov3.Utils.train_module import train_module


import pytorch_warmup as warmup
from warmup_scheduler import GradualWarmupScheduler
torch.autograd.set_detect_anomaly(True)


from Yolov3.Utils.args_train import get_args
from Yolov3.Dataset.dataset import yoloCoreDataset
from Yolov3.Utils.create_model import create_model

# logger callback:


def create_writer(log_saving_path):
  writer = SummaryWriter(log_dir=log_saving_path)
  print(f'Log file is saved at <{log_saving_path}>')
  #logfile = open(os.path.join(log_dir, 'log.txt'), 'w')
  return writer

def template_dataLoaderFunc(dataSet: yoloCoreDataset, args, labels):
    if labels is None:
        with open(args.labels, 'r') as f:
            labels = f.data()
            labels.pop(-1)

    trainLoader = DataLoader(dataSet(path=args.data, labels=labels, max_objects=18, split=args.split,
                                           debug=False, draw=False, argument=False, is_train=True),
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_worker,
                             drop_last=False
                             )
    # dont use img augment for val
    valLoader = DataLoader(dataSet(path=args.data, labels=labels, max_objects=18, split=args.split,
                                         debug=False, draw=False, argument=False, is_train=False),
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=args.num_worker,
                           drop_last=False
                           )
    return trainLoader, valLoader

def train(
    pwd_path,
    dataLoaderFunc, # callback returns [trainLoader, valLoader]
    modelLoaderFunc=None, # callback load different architect model
):
    ###### handle args #########
    args = get_args()
    ############################

    trainLoader, valLoader = dataLoaderFunc(args)

    print('Succesfully load dataset')
    
    num_steps = len(trainLoader) * args.epoch
    lr_scheduler = None
    warmup_scheduler = None

    ############### Loading Model ####################
    device = 'cuda' if not args.use_cpu else 'cpu'
    if not args.cfg and modelLoaderFunc is None:
        print(f'Succesfully load model with default config')
        yolo = create_model(num_classes=args.num_class,
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos,
                            device=device,
                            use_focal_loss=args.use_focal_loss
                            )
    elif args.cfg == 'default':
        f = os.path.join(pwd_path, 'config', 'yolov3.cfg')
        print(f"Succesfully load model with custom config at '{f}' ")
        yolo = create_model(None, default_cfg=f,
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos,
                            device=device,
                            use_focal_loss=args.use_focal_loss
                            )
    else:
        print(f"Succesfully load model with custom config at '{args.cfg}' ")
        yolo = create_model(None, default_cfg=args.cfg,
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos,
                            device=device,
                            use_focal_loss=args.use_focal_loss
                            )
    if not modelLoaderFunc is None:
        print('Load new architect with yoloHead ..')
        yolo = modelLoaderFunc(
            lb_noobj = args.lb_noobj,
            lb_obj = args.lb_obj,
            lb_class = args.lb_clss,
            lb_pos = args.lb_pos,
            device=device,
            use_focal_loss=args.use_focal_loss
            )
        print('Done, successfully loading model')
    
    ##############################################
    if args.log_path == "checkpoint":
        log_folder = os.path.join(pwd_path, args.log_path, 'yolo')
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
        if args.use_pretrained and args.freeze_backbone:
            grads_layer = ("81", "93", "105")
            for name, child in yolo.yolov3.named_children():
                if not name in grads_layer:
                    for each in child.parameters():
                        each.requires_grad = False
            print(f'Freezed all layers excludings {grads_layer}')
        if os.path.exists(log_file):
            old = datetime.now().strftime("%S%M%H_%d%m%Y") + "_old"
            shutil.copytree(log_file, os.path.join(
                log_folder, old), False, None)  # copy to old folder
            shutil.rmtree(log_file, ignore_errors=True)  # remove log
        lr_rate = args.lr
        wd = args.wd
        momen = args.momentum
        start_epoch = 1
        if args.optim == 'sgd':
            print(
                f"Use optimizer SGD with lr {lr_rate} wdecay {wd} momentum {momen}")
            optimizer = optim.SGD(
                yolo.parameters(), lr=lr_rate, weight_decay=wd, momentum=momen)
        elif args.optim == 'adam':
            print(f"Use optimizer Adam with lr {lr_rate} wdecay {wd} ")
            optimizer = optim.Adam(
                yolo.parameters(), lr=lr_rate, weight_decay=wd)
        else:
            raise f"optimizer {args.optim} is not supported"

        writer = create_writer(log_file)  
    #set up lr scheduler and warmup
    #num_steps = len(trainLoader) * args.epoch # move to line 174
    if not lr_scheduler and args.use_scheduler:
        print('Set up scheduler and warmup')
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)
        warmup_scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=4, after_scheduler=lr_scheduler)
    # Parallelism
    #if not args.use_cpu:
        #yolo = torch.nn.DataParallel(yolo)
    print('Start training model by GPU') if not args.use_cpu else print(
        'Start training model by CPU')
    train_module(
        model=yolo,
        trainLoader=trainLoader,
        valLoader=valLoader,
        optimizer_name=args.optim,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
        Epochs=args.epoch,
        use_cuda=not args.use_cpu,
        writer=writer,
        path=log_folder,
        lr_rate=lr_rate,
        wd=wd,
        momen=momen,
        start_epoch=start_epoch
    )
