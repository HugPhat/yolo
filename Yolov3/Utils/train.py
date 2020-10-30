import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

def unpack_data_loss_function(loss_accumulate, loss, writer, batch_index,checkpoint_index, mode, print_now:bool):
    """Accumulate loss value if variable exist else create dict element of [loss_accumulate] || Unpack precision || Write loss values to tensorboard
    

    Args:
        loss_accumulate ([dict]): [dictionary of all loss values]
        loss ([dict]): [loss values passing to loss function]
        writer ([SummaryWriter]): [tensorboard writer object]
        checkpoint_index ([int]): [checkpoint_index of update round]
        mode ([str]): [train or val]
        print_now([bool]): [execute print log]
    """
    ##
    
    loss_keys = list(loss.keys())
    for name_loss in loss_keys:
        if len(loss[name_loss]) == 1:
            try:
                loss_accumulate[name_loss + "/" + mode] += loss[name_loss]
            except KeyError:
                loss_accumulate.update( {name_loss + "/" + mode: loss[name_loss]})
        else:
            for sub in list(loss[name_loss].keys()):
                try:
                    loss_accumulate[name_loss + "@" + sub+ "/" + mode] += loss[name_loss][sub]
                except KeyError:
                    loss_accumulate.update( {name_loss + "@" + sub + "/" + mode: loss[name_loss][sub]})
    ##
    # epoch go from 1 to dataloader_len
    
    if print_now and writer:    
        for (name, loss_value) in loss_accumulate.items():
            if loss_value is not dict:
                writer.add_scalar(name, loss_value/batch_index,  checkpoint_index)
            else:
                for (pres, value) in loss_value.items():
                    writer.add_scalar(pres, value/batch_index, checkpoint_index)
    
    return loss_accumulate

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
    best_train_loss = np.inf
    best_val_loss = np.inf
    best_precision = 0
    best_recall =  0
    best_epoch = 0
    
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
        loss, lossX, lossY, lossW, lossH, lossConfidence, lossClass, recall, precision = 0, 0, 0, 0, 0, 0, 0, 0, 0
        with tqdm(total = len(trainLoader)) as epoch_pbar:
            epoch_pbar.set_description(f'[Train] Epoch {epoch}')
            
            for batch_index, (input_tensor, target_tensor) in enumerate(trainLoader):
                
                input_tensor = input_tensor.type(FloatTensor)
                target_tensor = target_tensor.type(FloatTensor)
                # zero grads
                optimizer.zero_grads()
                output = model(input_tensor, target_tensor) # return dictionary
                
                loss_set = loss_function(output)
                checkpoint_index = (epoch -1)*len(trainLoader) + batch_index
                print_now = True if (checkpoint_index + 1) % 20 == 0 else False
                unpack_data_loss_function(loss_accumulate, loss_set, writer, batch_index + 1, checkpoint_index, 'train', print_now)
                description = f'[Train: {epoch}/{Epochs} Epoch]: '
                for (name, loss_value) in loss_accumulate.items():
                    description += f"| {name.split('/')[0]} : {str(loss_value/ batch_index)} |"
                is_best_loss = loss_accumulate['loss'] / (batch_index +1)
                if is_best_loss < best_train_loss:
                    best_train_loss = is_best_loss
                    best_epoch = epoch
                    
                if lr_scheduler:
                    lr_scheduler.step(epoch-1)
                    #warmup_scheduler.dampen()
                
                loss_set['loss'].backward()
                optimizer.step()
                