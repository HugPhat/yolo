import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

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
        desc += '| ' + k
        for i, each in enumerate(v):
            desc += str(each/batch_index) + " |"
            temp.update({'layer_' + str(i) : each/batch_index})
        if print_now:
            if mode == 'val':
               writer.add_scalr(k + "/" + mode , temp, epoch)
            else:
                writer.add_scalr(k + "/" + mode, temp, epoch)

    return loss_accumulate, desc

def Train(
    model,  # yolov3
    trainLoader: DataLoader, # train: DataLoader
    valLoader : DataLoader, # val: DataLoader
    optimizer : optim, # optimizer 
    lr_scheduler,
    warmup_scheduler,
    loss_function, # Loss function
    writer, # tensorboard logger 
    use_cuda: bool,
    Epochs : int,
    
):
    """Template Train function 

    Args:
        model ([yolov3]): [pre defined yolov3 model]
        trainLoader([Dataloader])
        valLoader([Dataloader])
        optimizer([torch.optim])
        lr_scheduler
        warmup_schedule : [learning rate warmup]
        loss_function : [custom loss function]
        Epochs (int): [number of epoch]
        use_cuda (bool): use cuda (gpu) or not
    Returns:
        [type]: [description]
    """
    loss_value = 0
    # 
    accuracy_array = []
    recall_array = []
    precision_array = []
    if use_cuda:
        FloatTensor = torch.cuda.FloatTensor
        model.cuda()
    else:
        FloatTensor = torch.FloatTensor
    for epoch in range(1, Epochs + 1):
        model.train()
        loss_accumulate = {}
        #loss, lossX, lossY, lossW, lossH, lossConfidence, lossClass, recall, precision = 0, 0, 0, 0, 0, 0, 0, 0, 0
        with tqdm(total = len(trainLoader)) as epoch_pbar:
            epoch_pbar.set_description(f'[Train] Epoch {epoch}')
            
            for batch_index, (input_tensor, target_tensor) in enumerate(trainLoader):
                input_tensor = input_tensor.type(FloatTensor)
                target_tensor = target_tensor.type(FloatTensor)
                # zero grads
                optimizer.zero_grads()
                output = model(input_tensor, target_tensor) # return loss
                loss_value += output.item()
                checkpoint_index = (epoch -1)*len(trainLoader) + batch_index
                write_now =  (checkpoint_index + 1) % 20 == 0 
                loss_accumulate, desc = unpack_data_loss_function(
                    loss_accumulate, model.losses_log, writer, batch_index + 1, checkpoint_index, 'train', write_now, epoch)
                description = f'[Train: {epoch}/{Epochs} Epoch]: ||Total Loss: {loss_value/ (batch_index + 1)} ||->{desc}<-||'
                epoch_pbar.set_description(description)
                epoch_pbar.update(batch_index)
                if lr_scheduler:
                    lr_scheduler.step(epoch-1)
                    #warmup_scheduler.dampen()
                output.backward()
                optimizer.step()
        loss_accumulate = {}
        model.eval()
        with tqdm(total=len(valLoader)) as epoch_pbar:
            epoch_pbar.set_description(f'[Validate] Epoch {epoch}')
            for batch_index, (input_tensor, target_tensor) in enumerate(valLoader):
                input_tensor = input_tensor.type(FloatTensor)
                target_tensor = target_tensor.type(FloatTensor)    
                # return dictionary
                output = model(input_tensor, target_tensor)  # return loss
                checkpoint_index = batch_index + 1
                write_now = (batch_index + 1) == len(valLoader)
                loss_accumulate, desc = unpack_data_loss_function(
                    loss_accumulate, model.losses_log, writer, batch_index + 1, checkpoint_index, 'val', write_now, epoch)
                description = f'[Validate: {epoch}/{Epochs} Epoch]: ||->{desc}<-||'
                epoch_pbar.set_description(description)
                epoch_pbar.update(batch_index)
        
                
