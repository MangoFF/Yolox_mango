<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import argparse
import random
import warnings
from loguru import logger
import torch
import torch.backends.cudnn as cudnn
from yolox.core import Trainer, launch
from yolox.utils import configure_nccl, configure_omp, get_num_devices
import os
from yolox.exp import Exp as MyExp
from yolox.models import EfficientNet
class Exp(MyExp):
    """
    set the model detail here,inclue the data,dataaug etc.
    """
    def __init__(self,output_dir):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir="datasets/COCO/"
        #self.data_dir = "datasets/COCO/"
        self.output_dir = output_dir
        # yolox_l 不用很大的模型
        self.depth = 1
        self.width = 1
        size = 544
        lrd = 10
        self.multiscale_range = 0
        self.max_epoch = 64
        self.warmup_epochs = 10
        self.no_aug_epochs = 10
        self.num_classes = 10
        self.min_lr_ratio = 0.01

        self.input_size = (size, size)
        self.test_size = (size, size)
        self.basic_lr_per_img = 0.01 / (64.0 * lrd)
        self.eval_interval = 20

    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        return model

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )

    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")

    parser.add_argument(
        "-d", "--devices", type=int, default=1, help="device for training"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default="ckpt/yolox_l.pth", type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # this arg should be set in the huawei notebook
    parser.add_argument("--model",type=str,default="YOLOX_out",help='the path model saved')

    return parser

@logger.catch
def main(exp, args):
    # do not set the seed to speed up
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )
    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    # something set in the args some in the exp ,so we pass all of two
    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    # the model will be set in the huawei train notebook
    exp = Exp(output_dir=args.model)
    # this line merge the input into the exp,by overwrite
    exp.merge(args.opts)
    args.experiment_name="taril"

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )

