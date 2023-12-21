Install requirement - pip install -r requirements.txt
# Traffic sign detection and recognition

Our project is focused on recognizing traffic signs using data from the *Mapillary Traffic Sign Dataset*. Our main focus was on fine-tuning the YOLOv8 model, which tends to produce state-of-the-art results for many object detection tasks in real time. In principle, we employ three different approaches. The first approach involves a one-step process, utilizing a YOLOv8 model for simultaneous traffic sign detection and classification. The second approach employs two separate YOLOv8 models — one for binary detection (sign/no-sign) and another for classification of the pre-detected sign. The third approach involves fine-tuning the Object detection transformer DETR. See the documentation for further details.

## Data preprocessing 
- download dataset and edit the structure of it according to documentation
- edit PATH variable in data.py
- run the data.py script
  
## Model trainings 
- edit path and other constants at the beginning of files
- run 
  - model.py to train binary detector and yolo classifier models or yolo end to end detector and classifier model
  - train_cls.py to train simple CNN classifier
  - train_detr.py to train DETR model

## Evaluation
- run eval_double_step to run decoupled approach evaluation
- to run evaluation of simultanious approach, load model using *model = YOLO('path/to/best.pt')* and run *model.eval()*

## Inference with visualisation
- edit paths to image and models in inference.py
- run inference.py

## Acknowledgments and Links
> **Notice**: The complete GitHub repository exceeded the size limit of the assignment. Therefore, we are providing you with a link to access the repository hosted at https://github.com/.

- The GitHub repository is accessible [here](https://github.com/vtlustos/pova-traffic-sign-recognition.git)
- The fine-tuned models are available [here](https://huggingface.co/jkot/pova-traffic-sign-recognition-models)
- The code for the DETR was adapted from [this](https://colab.research.google.com/github/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb)
