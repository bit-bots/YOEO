# YOEO -- You Only Encode Once
A CNN for Embedded Object Detection and Semantic Segmentation

This project is based upon [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and will continuously be modified to our needs.


<img src="https://user-images.githubusercontent.com/15075613/131497667-c4e3f35f-db4b-4a53-b816-32dac6e1d85d.png" alt="example_image" height="180"/><img src="https://user-images.githubusercontent.com/15075613/131497744-e142c4ed-d69b-419a-96c3-39d871796081.png" alt="example_image" height="180"/><img src="https://user-images.githubusercontent.com/15075613/131498064-fc6545d9-8a1d-4953-a80b-52a3d2293c83.jpg" alt="example_image" height="180"/><img src="https://user-images.githubusercontent.com/15075613/131499391-e14a968a-b403-4210-b5f7-eb9be90a61db.png" alt="example_image" height="180"/>

<img src="https://user-images.githubusercontent.com/15075613/131497149-c35f315c-4bf8-44d2-a2a9-ce4012b946e7.png" alt="example_image" height="180"/><img src="https://user-images.githubusercontent.com/15075613/131497225-ce03abb5-52b0-4ee5-aeae-27816ba78996.png" alt="example_image" height="180"/><img src="https://user-images.githubusercontent.com/15075613/131495441-457d1290-c411-4e91-b1f6-3308cb89180d.jpg" alt="example_image" height="180"/><img src="https://user-images.githubusercontent.com/15075613/131499468-8ba66af0-e497-4a4f-a7f3-8db5efc47ce8.png" alt="example_image" height="180"/>

## Installation
### Installing from source

For normal training and evaluation we recommend installing the package from source using a poetry virtual environment.

```bash
git clone https://github.com/bit-bots/YOEO
cd YOEO/
pip3 install poetry --user
poetry install
```

You need to join the virtual environment by running `poetry shell` in this directory before running any of the following commands without the `poetry run` prefix.
Also have a look at the other installing method, if you want to use the commands everywhere without opening a poetry-shell.

#### Download pretrained YOLO weights

```bash
./weights/download_weights.sh
```

## Test
Evaluates the model on the test dataset.
See help page for more details.

```bash
poetry run yoeo-test -h
```

## Inference
Uses pretrained weights to make predictions on images.

```bash
poetry run yoeo-detect --images data/samples/
```

<p align="center"><img src="https://github.com/eriklindernoren/PyTorch-YOLOv3/raw/master/assets/giraffe.png" width="480"\></p>

## Train
For argument descriptions have a look at `poetry run yoeo-train --h`

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```bash
poetry run tensorboard --logdir='logs' --port=6006
```

Storing the logs on a slow drive possibly leads to a significant training speed decrease.

You can adjust the log directory using `--logdir <path>` when running `tensorboard` and `yolo-train`.

## Train on Custom Dataset

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### YOLO Annotation Folder
Move your yolo annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Segmentation Annotation Folder
Move your segmentation annotations to `data/custom/segmentations/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/segmentations/train.png`. The classes for each pixel are encoded via the class id.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```bash
poetry run yolo-train --model config/yoeo-custom.cfg --data config/custom.data
```

## API

You are able to import the modules of this repo in your own project if you install this repo as a python package.

An example prediction call from a simple OpenCV python script would look like this:

```python
import cv2
from yoeo import detect, models

# Load the YOLO model
model = models.load_model(
  "<PATH_TO_YOUR_CONFIG_FOLDER>/yolov3.cfg", 
  "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yolov3.weights")

# Load the image as a numpy array
img = cv2.imread("<PATH_TO_YOUR_IMAGE>")

# Convert OpenCV bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Runs the YOEO model on the image 
boxes, segmentation = detect.detect_image(model, img)

print(boxes)
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]

print(segmentation)
# Output will be a 2d numpy array with the coresponding class id in each cell
```

For more advanced usage look at the method's doc strings.

## Paper

The paper is currently in review.
