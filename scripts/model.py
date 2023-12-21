

import os
from ultralytics import YOLO

TRAIN_YOLO = True
TRAIN_DETR = False
DOUBLE_STEP = True
BASE_PATH = "../data/yolov8/detect"

detect_path = os.path.join(BASE_PATH, "dataset.yaml")
classify_path = os.path.join(BASE_PATH, "classify")

if TRAIN_YOLO and DOUBLE_STEP == False:
    model = YOLO("yolov8m.pt")  # load a pretrained model
    results = model.train(data=detect_path, epochs=100, imgsz=640, batch=16, fliplr=0)
    print(results)


if TRAIN_YOLO and DOUBLE_STEP:
    model = YOLO('yolov8n.pt')  # load a pretrained model
    results = model.train(data=detect_path, epochs=200, imgsz=640, save=True, workers=8, batch=16)
    print(results)

# 2. train sign classifier
if TRAIN_YOLO and DOUBLE_STEP:
    model = YOLO('yolov8x-cls.pt')  # load a pretrained model
    results = model.train(data=classify_path, epochs=100, imgsz=224, batch=128)
    print(results)


# DETR

PATH = 'C:/Users/tlust/Downloads/mtsd/coco'
NUM_CLASSES = 235
BATCH_SIZE = 6

import os
import torch
import torchvision
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
from pytorch_lightning import Trainer
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class MappilaryDataset(torchvision.datasets.CocoDetection):
    def __init__(self, dir, processor, train=True):
        super(MappilaryDataset, self).__init__(
            os.path.join(dir, "images"),
            os.path.join(dir, "annotations", "train.json" if train else "val.json")
        )
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(MappilaryDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {
            'image_id': image_id, 
            'annotations': target
        }
        encoding = self.processor(
            images=img, 
            annotations=target, 
            return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target
    
    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch  


# %%
class Detr(pl.LightningModule):
   def __init__(self, lr, lr_backbone, weight_decay, train_dl, val_dl):
      super().__init__()
      self.model = DetrForObjectDetection.from_pretrained(
         "facebook/detr-resnet-50",
         num_labels=NUM_CLASSES,
         ignore_mismatched_sizes=True
      )
      self.lr = lr
      self.lr_backbone = lr_backbone
      self.weight_decay = weight_decay
      self.train_dl = train_dl
      self.val_dl = val_dl
      self.map = MeanAveragePrecision()

   def forward(self, pixel_values, pixel_mask):
      outputs = self.model(
         pixel_values=pixel_values, 
         pixel_mask=pixel_mask
      )
      return outputs   
   
   def training_step(self, batch, batch_idx):
      # 1: forward pass
      outputs = self.model(
         pixel_values=batch["pixel_values"], 
         pixel_mask=batch["pixel_mask"],
         labels=[
            {k: v.to(self.device) for k, v in t.items()} 
            for t in batch["labels"]
         ]
      )

      # 2: log loss
      self.log("train_loss", outputs.loss)
      for k,v in outputs.loss_dict.items():
         self.log("train_" + k, v.item())

      # 3: backpropagation
      return outputs.loss

   def validation_step(self, batch, batch_idx):
      # 1: forward pass
      outputs = self.model(
         pixel_values=batch["pixel_values"], 
         pixel_mask=batch["pixel_mask"],
         labels=[
            {k: v.to(self.device) for k, v in t.items()} 
            for t in batch["labels"]
         ]
      )

      # 2: log loss
      self.log("val_loss", outputs.loss)
      for k,v in outputs.loss_dict.items():
         self.log("val_" + k, v.item())

      # 3: compute mAP
      preds = [
         {
            "boxes": outputs["pred_boxes"][i],
            "scores": torch.softmax(outputs["logits"][i], -1).max(-1).values,
            "labels": torch.softmax(outputs["logits"][i], -1).max(-1).indices
         } for i in range(len(outputs["pred_boxes"]))
      ]
      targets = [
         {
            "boxes": label["boxes"],
            "labels": label["class_labels"]
         } 
         for label in batch["labels"]
      ]
      self.map.update(preds, targets)
   
   def on_validation_epoch_end(self):
      self.log('val_mAP',self.map.compute()['map'], prog_bar=True)

   def configure_optimizers(self):
      param_dicts = [
            {
               "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
               "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
               "lr": self.lr_backbone,
            }
      ]
      optimizer = torch.optim.AdamW(
         param_dicts, 
         lr=self.lr,
         weight_decay=self.weight_decay
      )
      return optimizer 
   
   def train_dataloader(self):
      return self.train_dl
   
   def val_dataloader(self):
      return self.val_dl

# %%
if TRAIN_DETR:
  processor = DetrImageProcessor.from_pretrained(
      "facebook/detr-resnet-50"
  )

  # 1: load dataset
  train_dataset = MappilaryDataset(
      dir=PATH, 
      processor=processor
  )
  val_dataset = MappilaryDataset(
      dir=PATH, 
      processor=processor, train=False
  )
  train_dataloader = DataLoader(
    train_dataset, 
    collate_fn=train_dataset.collate_fn, 
    batch_size=BATCH_SIZE, 
    shuffle=True
  )
  val_dataloader = DataLoader(
    val_dataset, 
    collate_fn=train_dataset.collate_fn, 
    batch_size=BATCH_SIZE
  )

  # 2: init model
  model = Detr(
    lr=1e-4, 
    lr_backbone=1e-5,
    weight_decay=1e-4, 
    train_dl=train_dataloader, 
    val_dl=val_dataloader
  )

  # 3: fine-tune
  trainer = Trainer(
      devices=1, 
      accelerator="gpu",
      max_epochs=10, 
      gradient_clip_val=0.1, 
      accumulate_grad_batches=(32 // BATCH_SIZE), 
      log_every_n_steps=1
  )
  trainer.fit(model)


