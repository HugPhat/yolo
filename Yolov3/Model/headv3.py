from torch.autograd import Variable
import torch.nn as nn
import torch
from .Utils.utils import build_targets

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

        self.mse_loss = nn.MSELoss(size_average=True)
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ce_loss = nn.CrossEntropyLoss()

        self.confidence_points = [0.5, 0.75, 0.85]

    def calculate_precision(self, nCorrect, _tensor):
        precision = {}
        for each in self.confidence_points:
            nProposals = int((_tensor > each).sum().item())
            if nProposals > 0:
                precision.update({each: float(nCorrect/ nProposals)})
            else:
                precision.update({each: 0})

        return precision

    def forward(self, x, targets=None):

        isTrain = True if targets != None else False
        numBatches = x.size(0)
        numGrids = x.size(2)

        ratio = self.img_size / numGrids

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if x.is_cuda else torch.BoolTensor
        
        predict = x.view(numBatches, self.num_anchor,
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
        pred_boxes[..., 0] = x.data + coor_x

        pred_boxes[..., 1] = y.data + coor_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_x
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_y

        if isTrain:
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=conf.cpu().data,
                pred_cls=clss.cpu().data,
                target=targets.cpu().data,
                anchors=scale_anchors.cpu().data,
                num_anchors=self.num_anchor,
                num_classes=self.num_classes,
                grid_size=numGrids,
                ignore_thres=self.ignore_thres,
                img_dim=self.img_size,
            )

            recall = float(nCorrect / nGT) if nGT else 1

            precision = self.calculate_precision(nCorrect, conf)

            #print(f'proposals {nProposals} | correct {nCorrect} | nGT {nGT}')

            # Handle masks
            mask = Variable(mask.type(BoolTensor))
            conf_mask = Variable(conf_mask.type(BoolTensor))
            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask  # mask ^ conf_mask

            # Mask outputs to ignore non-existing objects
            giou = 1.0
            loss_x = self.mse_loss(x[mask], tx[mask])*giou
            loss_y = self.mse_loss(y[mask], ty[mask])*giou
            loss_w = self.mse_loss(w[mask], tw[mask])*giou
            loss_h = self.mse_loss(h[mask], th[mask])*giou
            loss_conf = self.bce_loss(conf[conf_mask_false], tconf[conf_mask_false])*self.lb_noobj  \
                + self.bce_loss(conf[conf_mask_true],
                                tconf[conf_mask_true])*self.lb_obj

            loss_cls = self.ce_loss(clss[mask], torch.argmax(tcls[mask], 1))

            loss = (loss_x + loss_y + loss_w + loss_h) *self.lb_pos + loss_conf + loss_cls*self.lb_class
            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
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
