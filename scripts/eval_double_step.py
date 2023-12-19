import os
import shutil

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import yaml


detector = YOLO('../binary_detector-nano-1280.pt')
classifier = YOLO('runs/classify/train15/weights/best.pt')

detect_path = '/storage/brno12-cerit/home/xvlasa15/mapilary/yolov8/detect-full/val/images'

labels_path = '/storage/brno12-cerit/home/xvlasa15/mapilary/yolov8/detect-full/val/labels'

#read dataset yaml
def read_dataset_yaml(file_path):
    with open(file_path, 'r') as file:
        dataset = yaml.safe_load(file)
    return dataset

dataset_yaml_path = '/storage/brno12-cerit/home/xvlasa15/mapilary/yolov8/detect-full/dataset.yaml'
det_names = read_dataset_yaml(dataset_yaml_path)["names"]

det_names = {v: k for k, v in det_names.items()}



def on_predict_batch_end(predictor):
    # Retrieve the batch data
    filename, image, _, _ = predictor.batch

    # filename without path and extension
    filename = [os.path.splitext(os.path.basename(f))[0] for f in filename] 

    # Ensure that image is a list
    image = image if isinstance(image, list) else [image]

    # Combine the prediction results with the corresponding frames
    predictor.results = zip(predictor.results, filename, image)

detector.add_callback("on_predict_batch_end", on_predict_batch_end)

#load labels to memory
labels = {}
for label in os.listdir(labels_path):
    with open(os.path.join(labels_path, label)) as f:
        lines = f.readlines()
        name = label.split('.')[0]
        boxes = []
        for line in lines:
            cls, x_center, y_center, w, h = line.strip().split(' ')
            x_center, y_center, w, h = map(float, [x_center, y_center, w, h])

            # Convert to absolute coordinates
            x1 = (x_center - w / 2)
            y1 = (y_center - h / 2)
            x2 = (x_center + w / 2)
            y2 = (y_center + h / 2)

            boxes.append([int(cls), x1, y1, x2, y2])
        labels[name] = boxes



results = detector(detect_path, stream=True, verbose=True)
mAP = MeanAveragePrecision()

for (result, file, frame) in results:
    boxes = result.boxes  
    label = labels[file]
    target= {"boxes": [], "labels": []}
    for box in label:
        cls, x1_rel, y1_rel, x2_rel, y2_rel = box
        x1_abs = x1_rel * frame.shape[1]
        y1_abs = y1_rel * frame.shape[0]
        x2_abs = x2_rel * frame.shape[1]
        y2_abs = y2_rel * frame.shape[0]
        target["boxes"].append([x1_abs, y1_abs, x2_abs, y2_abs])
        target["labels"].append(int(cls))
    
    # convert to tensors
    target["boxes"] = torch.tensor(target["boxes"])
    target["labels"] = torch.tensor(target["labels"])


    xyxy = boxes.xyxy.cpu().numpy()
    preds = {"boxes": [], "labels": [], "scores": []}
    for box in xyxy:
        #crop the frame, frame is Torch tensor
        x1, y1, x2, y2 = box
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        res = classifier(crop, verbose=False)
        for r in res:
            class_name = r.names[r.probs.top1]
            class_idx = det_names[class_name]

            preds["boxes"].append([x1, y1, x2, y2])
            preds["labels"].append(class_idx)
            preds["scores"].append(r.probs.top1conf)

    # convert to tensors
    preds["boxes"] = torch.tensor(preds["boxes"])
    preds["labels"] = torch.tensor(preds["labels"])
    preds["scores"] = torch.tensor(preds["scores"])

    
    mAP.update([preds], [target])


print(mAP.compute())
        
