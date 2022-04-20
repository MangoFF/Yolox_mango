# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
"""
from model_service.pytorch_model_service import PTServingBaseService
from yolox.exp import Exp as MyExp
import argparse
from PIL import Image
import numpy as np
from loguru import logger

import os
import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from tools.demo import Predictor
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir="datasets/COCO/"
        #self.data_dir = "datasets/COCO/"
        self.depth = 1
        self.width = 1
        self.input_size = (480, 480)
        self.test_size = (480, 480)
        self.num_classes=10
        self.test_conf = 0.001
        # nms threshold
        self.nmsthre = 0.65
    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        #freeze_module(model.backbone.backbone)
        return model
def get_model(model_path, **kwargs):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    exp = Exp()
    model = exp.get_model()
    ckpt_file = model_path
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    model.eval()
    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file=None,
        decoder=None,
        device=device,
        fp16=True,
        legacy=False,
    )
    return predictor


class PTVisionService(PTServingBaseService):
#class PTVisionService:
    def __init__(self, model_name,model_path):
        # 调用父类构造方法
        super(PTVisionService, self).__init__(model_name,model_path)
        # 调用自定义函数加载模型
        self.model = get_model(model_path)
        # 加载标签
        self.label = [0,1,2,3,4,5,6,7,8,9]
        self.img_size = 1024

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input_image_key = 'images'
        self.data = {}
        self.data['nc'] = 10
        self.data['names'] = ['lighthouse', 'sailboat', 'buoy', 'railbar', 'cargoship', 'navalvessels', 'passengership', 'dock', 'submarine', 'fishingboat']
        self.class_map = self.data['names']

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content).convert('RGB')
                preprocessed_data[k] = [image, file_name]
        # print(preprocessed_data)
        return preprocessed_data

    def _postprocess(self, data):

        return data

    def _inference(self, data):
        result = {}
        image1 = np.array(data['images'][0])
        outputs, img_info = self.model.inferenceImg(image1)
        outputs = outputs[0]

        bboxes = outputs[:, 0:4]

        # preprocessing: resize
        bboxes /= img_info["ratio"]

        result['detection_classes'] = []
        result['detection_scores'] = []
        result['detection_boxes'] = []

        for p, b in zip(outputs.tolist(), bboxes.tolist()):
            b = [b[1], b[0], b[3], b[2]]  # y1 x1 y2 x2
            result['detection_classes'].append(self.class_map[int(p[6])])

            result['detection_scores'].append(round(p[4]*p[5], 5))

            result['detection_boxes'].append([round(x, 3) for x in b])
        return result

if __name__=="__main__":
    data={}
    img= Image.open("assets/boat.jpg")
    data["images"]=[img]
    data["images"].append("mango")
    p=PTVisionService(model_name="yolox",model_path="./best_ckpt.pth")
    print(p._inference(data))
