from ultralytics import YOLO
from PIL import Image

from ultralytics.data.dataset import YOLODataset
import cv2
import os

DOUBLE_STEP = False
OUT_PATH = 'output.jpg'
SOURCE_PATH = './3klvNU_hcT_REMiHJjg4ew.jpg'

BINARY_DETECTOR_PATH = '../models/binary_detector-small-1280.pt'
CLASSIFIER_PATH = '../models/classifier_nano.pt'
E2EDETECTOR_PATH = '../models/e2e_medium_1920.pt'
DATASET_PATH = '../data/yolov8-onestep/detect/dataset.yaml'

modelBinaryDetector = YOLO(BINARY_DETECTOR_PATH)
modelClassifier = YOLO(CLASSIFIER_PATH)
e2eDetector = YOLO(E2EDETECTOR_PATH)

if DOUBLE_STEP:
    detectionResult = modelBinaryDetector.predict(SOURCE_PATH, agnostic_nms=True)
else:
    detectionResult = e2eDetector.predict(SOURCE_PATH, agnostic_nms=True) 

image = cv2.imread(SOURCE_PATH)
imageCopy = image.copy()

for result in detectionResult:
    boxes = result.boxes  
    print(result)
    for box in boxes:
        print(box)
        xyxy = box.xyxy.cpu().numpy()  
        x1, y1, x2, y2 = xyxy[0]
        if DOUBLE_STEP:
            # get sign from image
            sign = image[
                int(y1):int(y2), 
                int(x1):int(x2)
            ]

            classificationResult = modelClassifier(sign)
            prediction_class_num = int(classificationResult[0].probs.top1)
            prediction_class_name = classificationResult[0].names[prediction_class_num]
            prediction_class_name  = '--'.join(prediction_class_name.split('--')[1:-1])                    

        else:
            prediction_class_name = result.names[int(box.cls)]

        cv2.rectangle(imageCopy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # draw a rectangle on the image
        cv2.putText(imageCopy, prediction_class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save the image with bounding boxes
cv2.imwrite(OUT_PATH, imageCopy)