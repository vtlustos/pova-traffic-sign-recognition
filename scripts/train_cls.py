from operator import le
import os
import shutil
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning as L
from cls import Classifier

# Define the path to your dataset
dataset_path = '../mapilary/yolov8/classify-full/'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path,"train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path,"val"), transform=transform)

# missing =(set(train_dataset.classes).difference(set(val_dataset.classes)))
# for folder in missing:
#     os.makedirs(os.path.join(dataset_path,"val",folder), exist_ok=True)
#     # copy one image from train to val
#     shutil.copy(
#         os.path.join(dataset_path,"train",folder, os.listdir(os.path.join(dataset_path,"train",folder))[0]),
#         os.path.join(dataset_path,"val",folder, os.listdir(os.path.join(dataset_path,"train",folder))[0])
#     )


# Load your dataset




# Get class names
train_classes = train_dataset.classes
val_classes = val_dataset.classes
print(len(train_classes), len(val_classes), train_classes == val_classes)




# Define data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

# Instantiate your model
model = Classifier()

# Define your training loop
trainer = L.Trainer(
    max_epochs=10,
    devices=1, 
    accelerator="gpu"
)

# Train your model
trainer.fit(model, train_loader, val_loader)