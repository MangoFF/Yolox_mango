from exps.default.yolox_x import Exp as MyExp
import torch.distributed as dist
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir="datasets/COCO/"
    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model