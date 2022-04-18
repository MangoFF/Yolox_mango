from yolox.exp import Exp as MyExp
import torch.distributed as dist
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir="datasets/COCO/"
        #yolox_l 不用很大的模型
        self.depth = 1.0
        self.width = 1.0
        self.num_classes=10
        #慢启动预热网络
        self.warmup_epochs = 2
        #针对每张图片的basic learn rate
        self.basic_lr_per_img = 0.01 / 640.0
        self.max_epoch = 10
        #最后两个epoch不进行数据增强
        self.no_aug_epochs = 2
        #每隔5个epch验证一次
        self.eval_interval = 5
    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model