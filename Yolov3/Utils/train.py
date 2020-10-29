import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

def unpack_data_loss_function(loss_accumulate, loss, writer, index, mode):
    """Accumulate loss value if variable exist else create dict element of [loss_accumulate] || Unpack precision || Write loss values to tensorboard
    

    Args:
        loss_accumulate ([dict]): [dictionary of all loss values]
        loss ([dict]): [loss values passing to loss function]
        writer ([SummaryWriter]): [tensorboard writer object]
        index ([int]): [index of update round]
        mode ([str]): [train or val]
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
    
    keys = list(loss["precision"].keys())
    for each in keys:
        writer.add_scalar("precision_@" + keys + "/train", loss["precision"][each], index)
    
    

def Train(
    model,  # yolov3
    trainLoader: DataLoader, # train: DataLoader
    valLoader : DataLoader, # val: DataLoader
    optimizer : optim, # optimizer 
    lr_schedule,
    warmup_schedule,
    loss_function, # Loss function
    writer, # tensorboard logger 
    use_cuda: bool,
    num_epoch : int,
    
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
        num_epoch (int): [number of epoch]
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
    for epoch in range(1, num_epoch + 1):
        model.train()
        loss_accumulate = {}
        loss, lossX, lossY, lossW, lossH, lossConfidence, lossClass, recall, precision = 0, 0, 0, 0, 0, 0, 0, 0, 0
        with tqdm(total = len(trainLoader)) as epoch_pbar:
            epoch_pbar.set_description(f'[Train] Epoch {epoch}')
            
            for batch_index, (input_tensor, target_tensor) in enumerate(trainLoader):
                bidx = batch_index + 1
                
                input_tensor = input_tensor.type(FloatTensor)
                target_tensor = target_tensor.type(FloatTensor)
                # zero grads
                optimizer.zero_grads()
                output = model(input_tensor, target_tensor) # return dictionary
                
                loss_set = loss_function(output)
                
                # split loss type
                loss  += loss_set["loss"].item()
                lossX += loss_set["x"]
                lossY += loss_set["y"]
                lossW += loss_set["w"]
                lossH += loss_set["h"]
                lossClass += loss_set["class"]
                lossConfidence += loss_set["confidence"]
                precision += loss_set["precision"]
                recall += loss_set["recall"]
                
                is_best_loss = loss / bidx
                if is_best_loss < best_train_loss:
                    best_train_loss = is_best_loss
                    best_epoch = epoch
                                    
                writer.add_scalar('train/loss',         loss/bidx,           bidx)
                writer.add_scalar('LossX/train',        lossX/bidx,          bidx)
                writer.add_scalar('LossY/train',        lossY/bidx,          bidx)
                writer.add_scalar('LossW/train',        lossW/bidx,          bidx)
                writer.add_scalar('LossH/train',        lossH/bidx,          bidx)
                writer.add_scalar('LossConf/train',     lossConfidence/bidx, bidx)
                writer.add_scalar('LossClass/train',    lossClass/bidx,      bidx)
                writer.add_scalar('Precision/train',    precision/bidx,      bidx)
                writer.add_scalar('Recall/test',        recall/bidx,         bidx)