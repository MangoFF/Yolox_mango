import os
from yolox.exp import Exp as MyExp
import torch.distributed as dist
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir="datasets/COCO/"
        #yolox_s 不用很大的模型
        self.depth = 0.33
        self.width = 0.50
        self.num_classes=10
        #慢启动预热网络
        self.warmup_epochs = 2
        #针对每张图片的basic learn rate
        self.basic_lr_per_img = 0.01 / 64.0#64*5
        self.max_epoch = 30
        #最后两个epoch不进行数据增强
        self.no_aug_epochs = 5
        #每隔5个epch验证一次
        self.eval_interval = 5
        #不让他变化尺寸，似乎不太好，还是设置成1把
        self.multiscale_range = 1
    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model