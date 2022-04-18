#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
os.system("pip install loguru")
os.system("pip install thop")
os.system("pip install pycocotools")
os.system("pip install tensorboard")
os.system("pip install opencv_python")
os.system("pip install tqdm")
os.system("pip install ninja")
os.system("pip install tabulate")
os.system("pip install scikit-image")
os.system("pip install Pillow")

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

class Exp(MyExp):
    def __init__(self,output_dir):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir="/home/ma-user/modelarts/user-job-dir/model/datasets/COCO/"
        self.output_dir = output_dir
        #yolox_s 不用很大的模型
        self.depth = 0.33
        self.width = 0.50
        self.num_classes=10
        #慢启动预热网络
        self.warmup_epochs = 2
        #针对每张图片的basic learn rate
        self.basic_lr_per_img = 0.01 / 64.0#64*5
        self.max_epoch = 50
        #最后两个epoch不进行数据增强
        self.no_aug_epochs = 5
        #每隔5个epch验证一次
        self.eval_interval = 10
        #不让他变化尺寸，似乎不太好，还是设置成1把
        self.multiscale_range = 2
    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

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
    parser.add_argument("-b", "--batch-size", type=int, default=512, help="batch size")
    parser.add_argument(
        "-d", "--devices", type=int, default=1, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/yolo_mango.py ",
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default="/home/ma-user/modelarts/user-job-dir/model/ckpt/yolox_s.pth", type=str, help="checkpoint file")
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
        default=True,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=True,
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
    parser.add_argument("--model",type=str,default="",help='the path model saved')

    return parser

@logger.catch
def main(exp, args):
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

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    #exp = get_exp(args.exp_file, args.name)
    print("save model is "+str(args.model))
    exp=Exp(output_dir=args.model)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

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