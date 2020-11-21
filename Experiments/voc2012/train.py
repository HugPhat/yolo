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
    # @@ path to voc data
    required.add_argument('--data', action='store',
                        default=None,
                        dest='data',
                        required=True,
                        help='path to voc data folder')
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
                          help='lambda objectness lossfunciton')
    # lambda NoObj
    optional.add_argument('--lno', action='store',
                          default=0.5, type=float,
                          dest='lb_noobj',
                          help='lambda objectless lossfunciton')
    # lambda position
    optional.add_argument('--lpo', action='store',
                          default=1.0, type=float,
                          dest='lb_pos',
                          help='lambda position lossfunciton')
    # lambda class
    optional.add_argument('--lcl', action='store',
                          default=1.0, type=float,
                          dest='lb_clss',
                          help='lambda class lossfunciton')
    args = parser.parse_args()
    
    ###### handle args #########
    

    #path_2_root = r"E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012"
    path_2_root = args.data

    

    labels = data.readTxt(os.path.join(File_Path, 'config', 'class.names'))
    labels.insert(0, 0)# plus 0th: background
    trainLoader = DataLoader(data.VOC_data(path=path_2_root, labels=labels, max_objects=15,
                                        debug=False, draw=False, argument=True, is_train=True),
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=args.num_worker,
                             drop_last=False
                            )
    valLoader = DataLoader(data.VOC_data(path=path_2_root, labels=labels, max_objects=15,
                                           debug=False, draw=False, argument=True, is_train=False),
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_worker,
                             drop_last=False
                             )
   
    if not args.cfg:                             
        yolo = create_model(num_classes=args.num_class,
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos
                            )
    elif args.cfg == 'default':
        yolo = create_model(None, default_cfg=os.path.join(File_Path, 'config', 'yolov3.cfg'),
                            lb_noobj=args.lb_noobj,
                            lb_obj=args.lb_obj,
                            lb_class=args.lb_clss,
                            lb_pos=args.lb_pos
                            )
    else:
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
        checkpoint = torch.load(os.path.join(log_folder, "current_checkpoint.pth"))
        yolo.load_state_dict(checkpoint['state_dict'])
        optim_name = checkpoint["optimizer_name"]
        lr_rate = checkpoint['lr']
        wd = checkpoint['wd']
        momen = checkpoint['m'],
        start_epoch = checkpoint['epoch']
        if optim_name == 'sgd':
            optimizer = optim.SGD(yolo.parameters(), 
                        lr=lr_rate,
                        weight_decay=wd, 
                        momentum=momen
                        )
        else:
            optimizer = optim.Adam(yolo.parameters(),
                                lr=lr_rate,
                                weight_decay=wd,
                                )
        optimizer.load_state_dict(checkpoint['optimizer'])
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
    print('Start training model by GPU') if not args.use_cpu else print(
        'Start training model by CPU')
    train(
        model=yolo,
        trainLoader=trainLoader,
        valLoader=valLoader,
        optimizer_name=args.optim,
        optimizer=optimizer,
        lr_scheduler=None,
        Epochs=args.epoch,
        use_cuda= not args.use_cpu,
        writer=writer,
        path=log_folder,
        lr_rate=lr_rate,
        wd=wd,
        momen=momen,
        start_epoch=start_epoch
    )
