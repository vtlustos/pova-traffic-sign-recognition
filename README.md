Install requirement - pip install -r requirements.txt

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
# Inference with visualisation
- edit paths to image and models in inference.py
- run inference.py

