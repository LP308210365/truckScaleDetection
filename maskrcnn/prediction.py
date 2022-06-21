import time
from pathlib import Path
import cv2, json
import torch
from matplotlib import pyplot as plt
from d2go.utils.demo_predictor import DemoPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
from d2go.runner import GeneralizedRCNNRunner
from d2go.model_zoo import model_zoo
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

yaml_file = 'mask_rcnn_fbnetv3a_dsmask_C4.yaml'
runner = GeneralizedRCNNRunner()
cfg = runner.get_default_cfg()
cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
# load weights
cfg.MODEL.WEIGHTS = r'.model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
# Set training data-set path
cfg.DATASETS.TEST = ("my_segdataset_train",)

# predictor = DefaultPredictor(cfg)


model = runner.build_model(cfg)
checkpointer = runner.build_checkpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

predictor = DemoPredictor(model)

def seg(img):
    outputs = predictor(img)
    return outputs

if __name__ == "__main__":
    img1 = "/home/liupeng/sdb1/PythonProject/scaleDetection/data/CarSegmentation/PNGImages/1.png"
    img2 = "/home/liupeng/sdb1/PythonProject/scaleDetection/data/CarSegmentation/PNGImages/2.png"

    im1 = cv2.imread(img1)
    im2 = cv2.imread(img2)
    tensor1 = torch.from_numpy(im1)
    tensor2 = torch.from_numpy(im2)
    tensor = torch.cat((tensor1, tensor2),0)
    start = time.time()
    outputs = predictor(tensor)
    print(outputs)
    # end = time.time()
    # print("detect time:" + str(end - start) + "s/image")
    # v = Visualizer(im[:, :, ::-1],
    #                MetadataCatalog.get("my_segdataset_train"),
    #                scale=0.8,
    #                instance_mode=ColorMode.IMAGE
    # )
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Passing the predictions to CPU from the GPU
    # cv2.namedWindow("image",0)
    # cv2.imshow("image",v.get_image()[:, :, ::-1])
    # cv2.waitKey(20)