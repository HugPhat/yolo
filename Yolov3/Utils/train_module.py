import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

def save_model(name, model, optimizer, path, epoch, optim_name, lr_rate,wd,m, lr_scheduler):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_name': optim_name,
        'lr': lr_rate,
        'wd': wd,
        'm':m,
        'sche': lr_scheduler.state_dict() if lr_scheduler else None,
    }
    path = os.path.join(path, name)
    torch.save(checkpoint,  path)

def unpack_data_loss_function(loss_accumulate, loss, writer, batch_index,checkpoint_index, mode, print_now:bool, epoch):
    """Accumulate loss value if variable exist else create dict element of [loss_accumulate] || Unpack precision || Write loss values to tensorboard

    Args:
        loss_accumulate ([dict]): [dictionary of all loss values]
        loss ([dict]): [loss values passing to loss function]
        writer ([SummaryWriter]): [tensorboard writer object]
        checkpoint_index ([int]): [checkpoint_index of update round]
        mode ([str]): [train or val]
        print_now([bool]): [execute print log]
    Returns:
        loss_accumulate([dict]) : [change of loss_accumulate]
    """
    ##
    if loss_accumulate =={}:
        keys = list(loss[0].keys())
        for key in keys:
            loss_accumulate.update({key : []})
    for order, layer in enumerate(loss):
        for (k, v) in loss_accumulate.items():
            try:
                loss_accumulate[k][order] += layer[k]
            except:
                loss_accumulate[k].insert(order, layer[k])
    desc = ""
    for (k, v) in loss_accumulate.items():
        temp = {}
        if not k in ['loss_x', 'loss_y', 'loss_w', 'loss_h']:
            desc += '|' + str(k) + ": " + str(round(sum(v)/ (batch_index *3), 3))
        for i, each in enumerate(v):
            #desc += str(each/batch_index) + " |"
            temp.update({'layer_' + str(i) : each/batch_index})
            if print_now:
                if writer:
                    if mode == 'val':
                        writer.add_scalar(k + '_layer_' + str(i) + "/" + mode , each/ batch_index, epoch)
                    else:
                        writer.add_scalar(
                            k + '_layer_' + str(i) + "/" + mode, each / batch_index, checkpoint_index)
                else:
                    pass 

    return loss_accumulate, desc

def train_module(
    model,  # yolov3
    trainLoader: DataLoader, # train: DataLoader
    valLoader : DataLoader, # val: DataLoader
    optimizer_name:str, # 
    optimizer : optim, # optimizer 
    lr_scheduler: optim.lr_scheduler,
    warmup_scheduler ,
    writer, # tensorboard logger 
    use_cuda: bool,
    Epochs : int,
    path:str,
    lr_rate:float,
    wd:float,# weightdecay
    momen:float,
    start_epoch=1,
    
):
    """Template Train function 

    Args: \n
        model (yolov3): pre defined yolov3 model \n
        trainLoader(Dataloader) \n
        valLoader(Dataloader) \n
        optimizer(torch.optim) \n
        lr_scheduler \n
        Epochs (int): number of epoch \n
        use_cuda (bool): use cuda (gpu) or not \n
        path (str) : path to save model checkpoint
    Returns:
        type: description
    """

    #accuracy_array = []
    #recall_array = []
    #precision_array = []
    best_loss_value = 1000
    best_current_model = 'best_model.pth'
    
    if use_cuda:
        FloatTensor = torch.cuda.FloatTensor
        #model.cuda()
    else:
        FloatTensor = torch.FloatTensor
    for epoch in range(start_epoch, Epochs + 1):
        model.train()
        loss_value = 0
        loss_accumulate = {}
        #loss, lossX, lossY, lossW, lossH, lossConfidence, lossClass, recall, precision = 0, 0, 0, 0, 0, 0, 0, 0, 0
        with tqdm(total = len(trainLoader)) as epoch_pbar:
            epoch_pbar.set_description(f'[Train] Epoch {epoch}')
            
            for batch_index, (input_tensor, target_tensor) in enumerate(trainLoader):
                input_tensor = input_tensor.type(FloatTensor)
                target_tensor = target_tensor.type(FloatTensor)
                # zero grads
                optimizer.zero_grad()
                output = model(input_tensor, target_tensor) # return loss
                if  torch.isinf(output).any() or torch.isnan(output).any():
                    print(f'inp max {torch.max(input_tensor)} | min {torch.min(input_tensor)}')
                    print(f'tar max {torch.max(target_tensor)} | min {torch.min(target_tensor)}')
                loss_value += output.item()/3
                checkpoint_index = (epoch -1)*len(trainLoader) + batch_index
                write_now =  (checkpoint_index + 1) % 1 == 0  ## 1-> 20

                loss_accumulate, desc = unpack_data_loss_function(
                        loss_accumulate, model.losses_log, writer, batch_index + 1, checkpoint_index, 'train', write_now, epoch)
                #print(loss_accumulate['total'])
                description = f'[Train: {epoch}/{Epochs} Epoch]:[{desc}]'
                epoch_pbar.set_description(description)
                epoch_pbar.update(1)
                output.backward()
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step(epoch)
                if warmup_scheduler:   
                    warmup_scheduler.step(epoch)
        loss_accumulate = {}
        model.eval()
        with tqdm(total=len(valLoader)) as epoch_pbar:
            epoch_pbar.set_description(f'[Validate] Epoch {epoch}')
            for batch_index, (input_tensor, target_tensor) in enumerate(valLoader):
                input_tensor = input_tensor.type(FloatTensor)
                target_tensor = target_tensor.type(FloatTensor)    
                # return dictionary
                with torch.no_grad():
                    output = model(input_tensor, target_tensor)  # return loss
                checkpoint_index = (epoch -1)*len(valLoader) + batch_index
                write_now = (batch_index + 1) == len(valLoader) 

                loss_accumulate, desc = unpack_data_loss_function(
                        loss_accumulate, model.losses_log, writer, batch_index + 1, checkpoint_index, 'val', write_now, epoch)
                description = f'[Validate: {epoch}/{Epochs} Epoch]:[{desc}]'
                epoch_pbar.set_description(description)
                epoch_pbar.update(1)
        total= sum(loss_accumulate['total']) / (len(valLoader))
        if total < best_loss_value:
            if not path is None:
                save_model(model=model, path=path, name=best_current_model, 
                    epoch=epoch, optimizer=optimizer, optim_name=optimizer_name, lr_rate=lr_rate, wd=wd, m=momen, lr_scheduler=lr_scheduler)
        if not path is None:
            save_model(model=model, path=path, name="current_checkpoint.pth",
                   epoch=epoch, optimizer=optimizer, optim_name=optimizer_name, lr_rate=lr_rate, wd=wd, m=momen, lr_scheduler=lr_scheduler)
        
