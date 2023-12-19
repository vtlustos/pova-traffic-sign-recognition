from ultralytics import YOLO
from PIL import Image

from ultralytics.data.dataset import YOLODataset
import cv2
import os

DOUBLE_STEP = False
OUT_PATH = 'output.jpg'
SOURCE_PATH = './WMtHLqcqf6I4h5OOpad8kQ.jpg'

BINARY_DETECTOR_PATH = '/storage/brno12-cerit/home/xkotou06/POVa/pova-traffic-sign-recognition/models/binary_detector-small-1280.pt'
CLASSIFIER_PATH = '/storage/brno12-cerit/home/xkotou06/POVa/pova-traffic-sign-recognition/models/classifier_nano.pt'
E2EDETECTOR_PATH = '/storage/brno12-cerit/home/xkotou06/POVa/pova-traffic-sign-recognition/models/e2e_detector_medium_merged.pt'
DATASET_PATH = '/storage/brno12-cerit/home/xkotou06/POVa/pova-traffic-sign-recognition/data/yolov8-onestep/detect/dataset.yaml'

modelBinaryDetector = YOLO(BINARY_DETECTOR_PATH)
modelClassifier = YOLO(CLASSIFIER_PATH)
e2eDetector = YOLO(E2EDETECTOR_PATH)

# modelDetector.predict("https://www.youtube.com/watch?v=ct0zoti_sww", save=True, imgsz=1280, conf=0.5)

# dataset = YOLODataset(DATASET_PATH, imgsz=1280)

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
        print(f"detected box: {x1}, {x2}, {y1}, {y2} with class {box.cls}")
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

            #TODO remove, debugging
            print(prediction_class_name)
            print("other possibilities:")
            for prediction in classificationResult[0].probs.top5:
                print(f"{int(prediction)} : {classificationResult[0].names[int(prediction)]}")

        else:
            prediction_class_name = result.names[int(box.cls)]
            #TODO remove, debugging
            print(f"Box: {x1}, {x2}, {y1}, {y2}, class: {prediction_class_name}")


        cv2.rectangle(imageCopy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # draw a rectangle on the image
        cv2.putText(imageCopy, prediction_class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save the image with bounding boxes
cv2.imwrite(OUT_PATH, imageCopy)