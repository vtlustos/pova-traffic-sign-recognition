
# %%
import os
import numpy as np
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# %%
PATH = "../mapilary"

splits_path = os.path.join(PATH, "mtsd_v2_fully_annotated/splits")
images_path = os.path.join(PATH, "images")
annotations_path = os.path.join(PATH, "mtsd_v2_fully_annotated/annotations")

detect_path = os.path.join(PATH, "yolov8", "detect")
cls_path = os.path.join(PATH, "yolov8", "classify-full")
coco_path = os.path.join(PATH, "coco")

# %% [markdown]
# # YOLOv8

# %%
DOUBLE_STEP = True

# %% [markdown]
# ## Detection dataset
# For a single-step process, classify directly into the full taxonomy; if not, employ binary detection.

# %%
labels = []

# statistics
rejected = 0
total = 0
sign_distr = {}

# for split in ['train', 'val']:
#     print("Processing {} split...".format(split))

#     # 0. create output directories if not exists
#     out_dir = os.path.join(detect_path, split)
#     if not os.path.exists(out_dir):
#         os.makedirs(os.path.join(out_dir, "images"))
#         os.makedirs(os.path.join(out_dir, "labels"))

#     with open(os.path.join(splits_path, split + ".txt")) as f:
#         ids = f.readlines()

#     for id in tqdm(ids, total=len(ids)):
#         total += 1    

#         # 1. set and validate paths
#         id = id.strip()
#         img_path = os.path.join(images_path, f"{id}.jpg")
#         ann_path = os.path.join(annotations_path, f"{id}.json")
#         out_img_path = os.path.join(out_dir, "images", f"{id}.jpg")
#         out_ann_path = os.path.join(out_dir, "labels", f"{id}.txt")  

#         # 1.2. skip if image or annotation does not exists
#         if (not os.path.exists(img_path)) or (not os.path.exists(ann_path)):
#             rejected += 1
#             continue

#         # 2. read annotation 
#         with open(ann_path, 'r') as f:
#             ann = json.load(f)

#         # 2.1 skip if panormaic image
#         if ann['ispano'] == True:
#             rejected += 1
#             continue

#         # 3. create annotation file and copy image
#         is_empty = True
#         for obj in ann['objects']:
#             # 3.1 get label index
#             if DOUBLE_STEP:
#                 obj['label'] = 'traffic-sign'                    
#             else:
#                 if obj['label'] == 'other-sign':
#                     continue
#                 else:
#                     obj['label'] = '--'.join(obj['label'].split('--')[1:-1])                      
                
#             if obj['label'] not in labels:
#                 labels.append(obj['label'])
#             label = labels.index(obj['label'])

#             # 3.2 set sign distribution
#             sign_distr[label] = sign_distr[label] + 1 if label in sign_distr else 1

#             # 3.3 get bounding box
#             bbox = obj['bbox']
#             x_center = np.clip(((bbox['xmin'] + bbox['xmax']) / 2) / ann['width'], 0, 1)
#             y_center = np.clip(((bbox['ymin'] + bbox['ymax'] ) / 2) / ann['height'], 0, 1)
#             width = np.clip((bbox['xmax'] - bbox['xmin']) / ann['width'], 0, 1)
#             height = np.clip((bbox['ymax'] - bbox['ymin']) / ann['height'], 0, 1)
#             obj_ann = f"{label} {x_center} {y_center} {width} {height} \n"

#             # 3.4 write annotation
#             if width > 0.01 and height > 0.01:
#                 is_empty = False
#                 with open(out_ann_path, "a") as f:  
#                     f.write(obj_ann)
            
#             if is_empty == False:
#                 # 3.5. copy the image
#                 shutil.copy(img_path, out_img_path)
#             else:
#                 rejected += 1

# # 4. create dataset.yaml
# with open(os.path.join(detect_path, "dataset.yaml"), "a") as f:
#     f.write(f"path: {detect_path}\n")
#     f.write(f"train: {os.path.join('train', 'images')}\n")
#     f.write(f"val: {os.path.join('val', 'images')}\n")
#     f.write(f"names:\n")
#     for ix, label in enumerate(labels):
#         f.write(f"  {ix}: {label}\n")

# %% [markdown]
# ## Classification dataset
# If a two-stage pipeline is chosen, then construct a classification dataset that exclusively includes extracted signs, sorted by their respective labels.

# %%
if DOUBLE_STEP:
    labels = []

    # statistics
    rejected = 0
    total = 0
    sign_distr = {}


    for split in ['train', 'val', 'test']:
        print("Processing {} split...".format(split))
   
        with open(os.path.join(splits_path, split + ".txt")) as f:
            ids = f.readlines()

        for id in tqdm(ids, total=len(ids)):
            total += 1

            # 1. set and validate paths
            id = id.strip()
            img_path = os.path.join(images_path, f"{id}.jpg")
            ann_path = os.path.join(annotations_path, f"{id}.json")
    
            # 1.2. skip if image or annotation does not exists
            if (not os.path.exists(img_path)) or (not os.path.exists(ann_path)):
                rejected += 1
                continue

            # 2. load the image
            img = cv2.imread(img_path)

            # 3. extract traffic sign and create classification dataset                
            with open(ann_path, 'r') as f:
                ann = json.load(f)

            for obj in ann['objects']:
                # 3.1 skip if other-sign
                if obj['label'] == 'other-sign':
                    continue
                # remove --gN suffix
                # obj['label'] = '--'.join(obj['label'].split('--')[:-1])

                # 3.2 get sign path and create directory if not exists
                sign_dir = os.path.join(cls_path, split, obj['label'])
                if not os.path.exists(sign_dir):
                    os.makedirs(sign_dir)

                # 3.3 increment sign counter
                if obj['label'] not in labels:
                    labels.append(obj['label'])
                label = labels.index(obj['label'])            
                sign_distr[label] = sign_distr[label] + 1 if label in sign_distr else 1

                # 3.4 get bounding box
                sign = img[
                    int(obj['bbox']['ymin']):int(obj['bbox']['ymax']), 
                    int(obj['bbox']['xmin']):int(obj['bbox']['xmax'])
                ]

                # 3.4 save sign
                if sign.shape[0] > 0 and sign.shape[1] > 0 and sign.shape[2] > 0:
                    sign_path = os.path.join(sign_dir, f"{id}_{obj['key']}.jpg")
                    cv2.imwrite(sign_path, sign)

# %% [markdown]
# ## 2. view dataset statistics

# %%
print("Images - total: {}".format(total))
print("Images - rejected: {}".format(rejected) + " ({:.2f}%)".format(rejected / total * 100))
print("Signs: {}".format(np.sum(sign_distr.values())))
for ix, value in sign_distr.items():
    print(f"{labels[ix]}: {value}")
plt.bar(labels, sign_distr.values())

# %% [markdown]
# # COCO


# # %%
# from globox import AnnotationSet

# # %%
# # create dirs if not exists
# img_dir = os.path.join(coco_path, "images")
# ann_dir = os.path.join(coco_path, "annotations")
# if not os.path.exists(ann_dir) or not os.path.exists(img_dir):
#     os.makedirs(ann_dir)
#     os.makedirs(img_dir)

# for split in ['train', 'val']:
#     yolo_img_dir = os.path.join(detect_path, split, "images")

#     # copy images
#     for img_name in os.listdir(yolo_img_dir):
#         shutil.copy(
#             os.path.join(yolo_img_dir, img_name),
#             os.path.join(img_dir, img_name)
#         )

#     # convert annotations
#     yolo = AnnotationSet.from_yolo_v5(
#         folder=os.path.join(detect_path, split, "labels"),
#         image_folder=yolo_img_dir
#     )
#     yolo.save_coco(os.path.join(ann_dir, split + ".json"), auto_ids=True)


