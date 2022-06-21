import os
import json
import numpy as np
import cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from d2go.model_zoo import model_zoo
from d2go.utils.demo_predictor import DemoPredictor
import matplotlib.pyplot as plt

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], evaluator_type="coco")
# balloon_metadata = MetadataCatalog.get("balloon_train")



import random



# for d in ["train", "val"]:
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], evaluator_type="coco")
register_coco_instances("my_segdataset_train", {}, "/home/liupeng/sdb1/Image/ScaleDetection/Train/carSeg/V2/Annotations/coco_info.json",
                        "/home/liupeng/sdb1/Image/ScaleDetection/Train/carSeg/V3/AugImages")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
from d2go.runner import GeneralizedRCNNRunner

def prepare_for_launch():
    runner = GeneralizedRCNNRunner()
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("mask_rcnn_fbnetv3a_dsmask_C4.yaml"))
    cfg.MODEL_EMA.ENABLED = False
    cfg.DATASETS.TRAIN = ("my_segdataset_train",)
    cfg.DATASETS.TEST = ("my_segdataset_train",)
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("mask_rcnn_fbnetv3a_dsmask_C4.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 6000    # 600 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg, runner

cfg, runner = prepare_for_launch()
model = runner.build_model(cfg)
runner.do_train(cfg, model, resume=False)


# import copy
# from detectron2.data import build_detection_test_loader
# from d2go.export.api import convert_and_export_predictor
# from d2go.utils.testing.data_loader_helper import create_fake_detection_data_loader
# from d2go.export.d2_meta_arch import patch_d2_meta_arch
#
# import logging
#
# # disable all the warnings
# previous_level = logging.root.manager.disable
# logging.disable(logging.INFO)
#
# patch_d2_meta_arch()
#
# cfg_name = 'mask_rcnn_fbnetv3a_dsmask_C4.yaml'
# pytorch_model = model_zoo.get(cfg_name, trained=True)
# pytorch_model.cpu()
#
# with create_fake_detection_data_loader(224, 320, is_train=False) as data_loader:
#     predictor_path = convert_and_export_predictor(
#             model_zoo.get_config(cfg_name),
#             copy.deepcopy(pytorch_model),
#             "torchscript_int8@tracing",
#             './',
#             data_loader,
#         )

# # recover the logging level
# logging.disable(previous_level)
#
# from mobile_cv.predictor.api import create_predictor
# model = create_predictor(predictor_path)
#
#
#
# from d2go.utils.demo_predictor import DemoPredictor
# im = cv2.imread("/home/liupeng/sdb1/PythonProject/scaleDetection/detectron2go/datasets/humandetect/images/train/20.png")
#
# predictor = DemoPredictor(model)
# outputs = predictor(im)
#
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("my_segdataset_train"))
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.imshow(out.get_image()[:, :, ::-1])
#
