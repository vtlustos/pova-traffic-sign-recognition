from ultralytics import YOLO
import cv2

MODEL_PATH = '16workers/runs/detect/train3/weights/best.pt'
OUT_PATH = 'output.jpg'
SOURCE_PATH = '../data/mtsd/images/X6vgO71T3_Yw8Vz_w_ldAw.jpg'


model = YOLO(MODEL_PATH)
results = model(SOURCE_PATH)  
image = cv2.imread(SOURCE_PATH)
for result in results:
    boxes = result.boxes  
    for box in boxes:
        xyxy = box.xyxy.cpu().numpy()  
        x1, y1, x2, y2 = xyxy[0]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # draw a rectangle on the image
        class_name = result.names[int(box.cls)]
        class_name  = '--'.join(class_name.split('--')[1:-1])                    
        cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save the image with bounding boxes
cv2.imwrite(OUT_PATH, image)