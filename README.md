**Re-implementation yolov3 in pytorch** <br>
**Extra dependencies**:
* install warmup <br>
http://github.com/ildoonet/pytorch-gradual-warmup-lr/master

**Usage**<br>
```
    train.py [-h] [--fr] [--pre] [--con] [--cfg CFG] [--ncl NUM_CLASS] 
                [--sch] --data DATA [--lb LABELS] [--split SPLIT] 
                [--bs BATCH_SIZE] [--nw NUM_WORKER] [--op {sgd,adam}] 
                [--mo MOMENTUM] [--lr LR] [--wd WD] [--ep EPOCH] [--cpu] 
                [--log LOG_PATH] [--lo LB_OBJ] [--lno LB_NOOBJ] [--lpo LB_POS] 
                [--lcl LB_CLSS] 
    
    optional arguments:
    -h, --help       show this help message and exit

    required arguments:
    --data DATA      path to data folder 
    --lb LABELS      path to labels

    optional arguments:
    --fr             freeze pretrained backbone (True)
    --pre            use pretrained (False)
    --con            not continue training
    --cfg CFG        use custom config, if use, pass the path of custom cfg
                    file, default is (./config/yolov3.cfg)
    --ncl NUM_CLASS  number of annot classes (21)
    --sch            set it to turn on using scheduler (False)
    --split SPLIT    split ratio [0., 1.] of voc dataset (None) if not None
    --bs BATCH_SIZE  number of batch size (8)
    --nw NUM_WORKER  number of worker (0)
    --op {sgd,adam}  type of optimizer: sgd/adam (sgd)
    --mo MOMENTUM    Momentum for sgd (0.91)
    --lr LR          learning rate (0.01)
    --wd WD          weight decay (1e-4)
    --ep EPOCH       number of epoch (20)
    --cpu            use cpu or not (False)
    --log LOG_PATH   path to save chkpoint and log (./checkpoint)
    --lo LB_OBJ      lambda objectness lossfunciton (2.0)
    --lno LB_NOOBJ   lambda objectless lossfunciton (0.5)
    --lpo LB_POS     lambda position lossfunciton (1.)
    --lcl LB_CLSS    lambda class lossfunciton (1.)
```