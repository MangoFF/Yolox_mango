# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
"""
from model_service.pytorch_model_service import PTServingBaseService

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
def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser

def get_model(model_path, **kwargs):
    batch_size = 1

    save_dir = ''

    if torch.cuda.is_available():
        device = torch.device('cuda')

    else:
        device = torch.device('cpu')
    dir_path = os.path.dirname(os.path.realpath(model_path))
    exp = get_exp(os.path.join(dir_path, "exps/example/yolo_mango.py"), "yolox_x")
    model = exp.get_model()
    logger.info(
        "Model Summary: {}".format(get_model_info(model, (640, 640)))
    )

    ckpt_file = model_path
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    model.eval()
    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        device, False, False,
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
    p=PTVisionService(model_name="yolox",model_path="./latest_ckpt.pth")
    print(p._inference(data))
