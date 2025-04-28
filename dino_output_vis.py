import torch
import cv2
import numpy as np
# import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# from dinov2.model import vit_giant2

model_weights = "/home/FOCUS/ckpt/dinov2_vitg14_pretrain_updated.pkl"
input_image_path = "/home/FOCUS/datasets/mvtec_good/images/0006.png"
output_image_path = "/home/FOCUS/datasets/mvtec_good/output/0006.png"

image = cv2.imread(input_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cfg = get_cfg()
cfg.merge_from_file("/home/FOCUS/configs/sod_dinov2_giant.yaml")
cfg.MODEL.WEIGHTS = model_weights
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite(output_image_path, cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR))

print(f"image saved: {output_image_path}")
