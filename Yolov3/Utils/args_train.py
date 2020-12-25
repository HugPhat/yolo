import argparse
def get_args():
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
    # @@ path to voc labels
    required.add_argument('--lb', action='store',
                          default=None,
                          dest='labels',
                          required=False,
                          help='path to labels')
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
    # use focal loss function
    optional.add_argument('--focal', action='store_true',
                          default=False,
                          dest='use_focal_loss',
                          help='use focal loss in clasification (false)')
    args = parser.parse_args()

    return args
