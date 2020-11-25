from torch.autograd import Variable
import torch.nn as nn
import torch
from Yolov3.Utils.utils import build_targets

class yoloHeadv3(nn.Module):
    def __init__(self, anchor,
                 num_classes,
                 img_size,
                 lb_noobj=1.0,
                 lb_obj=1.0,
                 lb_class=1.0,
                 lb_pos=1.0,
                 ignore_thresh = 0.5,
                 ):
        """Yolo detection layer

        Args:
            anchor ([list]): [list of anchor box]
            num_classes ([int]): [number of training classes]
            img_size ([int]): [fixed size image]
            lb_noobj (float, optional): [No function at this moment]. Defaults to 1.0.
            lb_obj (float, optional): [No function at this moment]. Defaults to 1.0.
            lb_class (float, optional): [No function at this moment]. Defaults to 1.0.
            lb_pos (float, optional): [No function at this moment]. Defaults to 1.0.
            ignore_thresh (float, optional): [threshold of objecness score]. Defaults to 0.5.
        """
        super(yoloHeadv3, self).__init__()
        self.anchor = anchor
        self.num_anchor = len(self.anchor)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_info = num_classes + 5

        self.ignore_thres = ignore_thresh
        self.lb_noobj = lb_noobj
        self.lb_obj = lb_obj
        self.lb_class = lb_class
        self.lb_pos = lb_pos

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.confidence_points = [0.5, 0.75, 0.85]

    def calculate_precision(self, nCorrect, _tensor) -> dict:
        """
        Returns:
            * {rangeA : value , rangeB: value, ...}
        """
        precision = {}
        for each in self.confidence_points:
            nProposals = int((_tensor.cpu().data > each).sum())
            if nProposals > 0:
                precision.update({ 'P'+str(each): float(nCorrect/ nProposals)})
            else:
                precision.update({'P'+str(each): 0})

        return precision

    def forward(self, X, targets=None):
        """[Feed forward function]

        Args:
            X ([tensor]): [input tensor shape (batch_size, 3, img_size, img_size)]
            targets ([tensor], optional): [tensor shape (batch_size, max_objects, 5 + number of classes)]. Defaults to None.

        Returns:
            [{
                "mask"      : mask,
                "x"         : [x, tx],
                "y"         : [y, ty],
                "w"         : [w, tw],
                "h"         : [h, th],
                "conf"      : [conf, tconf],
                "class"     : [clss, tcls],
                "recall"    : recall,
                "precision" : precision
            }]: [Pairs of value to calculate loss function]
        """
        
        isTrain = True if targets != None else False
        numBatches = X.size(0)
        numGrids = X.size(2)

        ratio = self.img_size / numGrids

        FloatTensor = torch.cuda.FloatTensor if X.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if X.is_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if X.is_cuda else torch.BoolTensor
        
        predict = X.view(numBatches, self.num_anchor,
                         self.grid_info, numGrids, numGrids).permute(0, 1, 3, 4, 2).contiguous()  #

        x = torch.sigmoid(predict[..., 0])
        y = torch.sigmoid(predict[..., 1])
        w = predict[..., 2]
        h = predict[..., 3]

        conf = torch.sigmoid(predict[...,   4])
        clss = torch.sigmoid(predict[...,   5:])
        #print(clss)
        coor_x = torch.arange(numGrids).repeat(numGrids, 1).view(
            [1, 1, numGrids, numGrids]).type(FloatTensor)
        coor_y = torch.arange(numGrids).repeat(numGrids, 1).t().view(
            [1, 1, numGrids, numGrids]).type(FloatTensor)
        scale_anchors = FloatTensor([(aw/ratio, ah/ratio)
                                     for aw, ah in self.anchor])
        anchor_x = scale_anchors[:, 0:1].view((1, self.num_anchor, 1, 1))
        anchor_y = scale_anchors[:, 1:2].view((1, self.num_anchor, 1, 1))

        pred_boxes = FloatTensor(predict[..., :4].shape)
        pred_boxes[..., 0] = x + coor_x

        pred_boxes[..., 1] = y + coor_y
        pred_boxes[..., 2] = torch.exp(w).clamp(max=1E3) * anchor_x
        pred_boxes[..., 3] = torch.exp(h).clamp(max=1E3) * anchor_y

        if torch.isnan(conf).any():
            print(f'predict { predict } size {predict.size()}')
            print(clss)
            print(predict[..., 4])
        if isTrain:
            #if x.is_cuda:
            #    self.mse_loss.cuda()
            #    self.ce_loss.cuda()
            #    self.bce_loss.cuda()

            nGT, nCorrect, mask, noobj_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes,
                pred_conf=conf,
                pred_cls=clss,
                target=targets,
                anchors=scale_anchors,
                num_anchors=self.num_anchor,
                num_classes=self.num_classes,
                grid_size=numGrids,
                ignore_thres=self.ignore_thres,

            )

            recall = float(nCorrect / nGT) if nGT else 1
            precision = self.calculate_precision(nCorrect, conf)

            # Handle masks
            mask = Variable(mask.type(BoolTensor))
            noobj_mask = Variable(noobj_mask.type(BoolTensor))
            
            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)


            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(conf[mask], tconf[mask])*self.lb_obj +\
                            self.bce_loss(conf[noobj_mask], tconf[noobj_mask])*self.lb_noobj

            loss_cls = self.ce_loss(clss[mask], torch.argmax(tcls[mask], 1))
            loss_pos = (loss_x + loss_y + loss_w + loss_h) 
            loss = loss_pos * self.lb_pos + loss_conf + loss_cls*self.lb_class
            
            if torch.sum(mask) == 0 or torch.isnan(loss_conf) or torch.isinf(loss_conf):
                print(
                    f'Target tensor max {torch.max(targets)} | min {torch.min(targets)}')
            
            return (
                loss,
                {
                    "total"  : loss.item(),
                    "loss_pos": loss_pos.item(),
                    "loss_x" : loss_x.item(),
                    "loss_y" : loss_y.item(),
                    "loss_w" : loss_w.item(),
                    "loss_h" : loss_h.item(),
                    "loss_conf" : loss_conf.item(),
                    "loss_class" : loss_cls.item(),
                    "recall" : recall,
                    **precision
                }
            )
        
        else:
            output = torch.cat(
                (
                    pred_boxes.view(numBatches, -1, 4)*ratio,
                    conf.view(numBatches, -1, 1),
                    clss.view(numBatches, -1, self.num_classes),
                ), -1,
            )

            return output
