#coding=utf-8
import math
import time
import os
import numpy as np
import torch
import cv2
from yolov5.utils.augmentations import letterbox
from d2go.runner import GeneralizedRCNNRunner
from d2go.model_zoo import model_zoo
from d2go.utils.demo_predictor import DemoPredictor
import socket

#yolo
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size

def box_in_mask_predict(outputs_mask, outputs_box, IOU):
    mask_instances = outputs_mask["instances"].to("cpu")
    masks_scores = mask_instances.scores if mask_instances.has("scores") else None
    masks = mask_instances.pred_masks.numpy() if mask_instances.has("pred_masks") else None
    mask = np.zeros(mask_instances.image_size)

    #chose main mask
    if len(masks) > 0:
        max_index = torch.argmax(masks_scores)
        if(masks_scores[max_index] > 0.9):
            mask = masks[max_index]

    box_instances = outputs_box.to("cpu")
    # boxes = box_instances.pred_boxes if box_instances.has("pred_boxes") else None
    boxes = box_instances.numpy().astype(np.int)
    # boxes_scores = box_instances.scores if box_instances.has("scores") else None
    # #delete error boxes
    # index_delete = np.where(boxes_scores < 0.45)
    # boxes = np.delete(boxes, index_delete, axis=0)

    #obtain boxes in car
    if boxes.shape[0] > 0:
        #boxes index in car judgement
        index_delete = index_foot_not_in_car(mask,boxes,IOU)

        #delete boxes not in car
        boxes = np.delete(boxes, index_delete, axis=0)

    return boxes


def index_foot_not_in_car(mask, boxes, IOU):
    index = []
    for i, box in enumerate(boxes):
        foot_img = np.zeros(mask.shape).astype(np.uint8)
        # height of foot(default: 0.2y of box)
        height_foot = int((box[3] - box[1])*0.3)
        box_foot = [box[0], box[3]-height_foot, box[2], box[3]]
        foot_img[box[3]-height_foot: box[3], box[0]: box[2] + 1] = True
        # plt.imshow(foot_img)
        # plt.show()
        if not_in_car_judement(mask,box_foot, foot_img, IOU):
            index.append(i)
    return index


def not_in_car_judement(mask, box_foot, foot_img, IOU):
    # plt.imshow(box_img)
    # plt.show()
    IOU_img = mask*foot_img
    # plt.imshow(IOU_img)
    # plt.show()
    IOU_pixels_num = np.sum(IOU_img == 1)

    IOU_real = IOU_pixels_num / ((box_foot[2] - box_foot[0])*(box_foot[3] - box_foot[1]))
    # if(IOU_real > IOU):
    #     print(IOU_real)
    if(IOU_real <= IOU or math.isnan(IOU_real)):
        return True
    return False


def transfer(box):
    x0 = box[0]
    y0 = box[1]
    x1 = box[2]
    y1 = box[3]
    if(x0 > x1):
        temp = x0
        x0 = x1
        x1 = temp
    if(y0 > y1):
        temp = y0
        y0 = y1
        y1 = temp
    return np.array((x0, y0, x1, y1))


def getPredictor(method, weight, threshould=0.5):
    runner = GeneralizedRCNNRunner()
    cfg = runner.get_default_cfg()
    if (method == "mask"):
        yaml_file = 'mask_rcnn_fbnetv3a_dsmask_C4.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
    elif(method == "box"):
        # BOX_PREDICT_PATH = "./fasterrcnn/torchscript_int8@tracing"
        imgsz = check_img_size(640, s=64)
    else:
        raise ValueError("method name error")

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    # load weights
    cfg.MODEL.WEIGHTS = weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshould  # set the testing threshold for this model

    if (method == "mask"):
        model = runner.build_model(cfg)
        checkpointer = runner.build_checkpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        predictor = DemoPredictor(model)
    elif (method == "box"):
        # model = create_predictor(BOX_PREDICT_PATH)
        # predictor = DemoPredictor(model)
        predictor =  DetectMultiBackend(weight, device=torch.device("cuda:0"))
    else:
        raise ValueError("method name error")

    return predictor


def imgAug(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    for i in range(image.shape[2]):
        image[:,:,i] = clahe.apply(image[:,:,i])
    return image


def frame_to_img(frame):
    img = letterbox(frame)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CWH, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(torch.device('cuda:0'))
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img


def in_car_judgement(outputs_mask, outputs_box, IOU=0.7):
    # human in car judgement
    # IOU 判定人是否在汽车衡上的iou
    boxes_in_car = box_in_mask_predict(outputs_mask, outputs_box, IOU)
    if (len(boxes_in_car) > 0):
        return True
    return False


if __name__ == "__main__":
    path = "/home/liupeng/sdb1/Image/ScaleDetection/Train/humanDetec/V4/images/12_09_21_02_07.jpg"
    src = cv2.imread(path)
    cv2.namedWindow("input image")
    cv2.imshow("input image", src)
    img = imgAug(src)
    cv2.namedWindow("aug image")
    cv2.imshow("aug image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()