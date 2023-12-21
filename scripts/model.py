# YOLO

import os
from ultralytics import YOLO

TRAIN_YOLO = True
TRAIN_DETR = False
DOUBLE_STEP = True
BASE_PATH = "../data/yolov8/detect"

detect_path = os.path.join(BASE_PATH, "dataset.yaml")
classify_path = os.path.join(BASE_PATH, "classify")

if TRAIN_YOLO and DOUBLE_STEP == False:
    # train a single model
    model = YOLO("yolov8m.pt")  # load a pretrained model
    results = model.train(data=detect_path, epochs=100, imgsz=640, batch=16, fliplr=0)
    print(results)

if TRAIN_YOLO and DOUBLE_STEP:
   # 1. train binary detector
    model = YOLO('yolov8n.pt')  # load a pretrained model
    results = model.train(data=detect_path, epochs=200, imgsz=640, save=True, workers=8, batch=16)
    print(results)

if TRAIN_YOLO and DOUBLE_STEP:
   # 2. train sign classifier
    model = YOLO('yolov8x-cls.pt')  # load a pretrained model
    results = model.train(data=classify_path, epochs=100, imgsz=224, batch=128)
    print(results)
