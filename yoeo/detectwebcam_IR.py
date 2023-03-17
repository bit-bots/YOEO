#! /usr/bin/env python3
from __future__ import division

import os
import argparse
import tqdm
import numpy as np
import cv2
import time

import sys
import time
from pathlib import Path
from openvino.runtime import Core


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from yoeo.models import load_model
from yoeo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info, rescale_segmentation
from yoeo.utils.datasets import ImageFolder
from yoeo.utils.transforms import Resize, DEFAULT_TRANSFORMS

import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def detect_directory(model_path, weights_path, classes, output_path,
                     batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    device = "CPU"
    model_xml_path = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/IR&onnx_for_416_Petr_1/yoeo.xml"
    ie = Core()
    ie.set_property({'CACHE_DIR': '../cache'})
    model = ie.read_model(model_xml_path)
    compiled_model = ie.compile_model(model=model, device_name=device)
    print(compiled_model.input(0).shape, compiled_model.output(1).shape)
    input_key = compiled_model.input(0)
    network_input_shape = list(input_key.shape)
    network_image_height, network_image_width = network_input_shape[2:]
    
    print("NY PRIVET 7")
    cam = cv2.VideoCapture(0)
    key = cv2.waitKey(1)

    while key != 27:
        print("START OF INFERENCE OF IMAGE")
        t = time.time()
        _, image = cam.read()

        fps = int(cam.get(cv2.CAP_PROP_FPS))
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        cv2.imshow('raw', image)

        result_image = detect_image(
            compiled_model,
            image,
            network_image_height,
            network_image_width,
            conf_thres,
            nms_thres)
        
        cv2.imshow("rawIMG", image)
        cv2.imshow("resultIMG", result_image)

        if cv2.waitKey(1) == 27:
            cam.release()
            cv2.destroyAllWindows()
            break

def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def to_bgr(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())

def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def detect_image(compiled_model, image, network_image_height=416, network_image_width=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class], Segmentation as 2d numpy array with the coresponding class id in each cell
    :rtype: nd.array, nd.array
    """
    # Resize to input shape for network.
    
    # resized_image = to_rgb(cv2.resize(src=image, dsize=(network_image_height, network_image_width)))

    # resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))

    # Reshape the image to network input shape NCHW.
    # input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    
    input_image = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(network_image_height)])((
            image,
            np.empty((1, 5)),
            np.empty((network_image_height, network_image_height), dtype=np.uint8)))[0].unsqueeze(0)

    print(f"raw image shape: {image.shape}")
    print(f"torch image shape: {input_image.shape}")
    print(f"torch image shape: {type(input_image)}")
    

    # if torch.cuda.is_available():
    #     input_img = input_img.to("cuda")
    print(f"model: {compiled_model}")
    # Get detections
    output_seg = compiled_model.output(1)
    result = compiled_model([input_image])[output_seg]

    # Convert the network result of disparity map to an image that shows
    # distance as colors.
    result_image = convert_result_to_image(result=result)
    # result_image = to_bgr(result_image)
    # Resize back to original image shape. The `cv2.resize` function expects shape
    # in (width, height), [::-1] reverses the (height, width) shape to match this.
    result_image = cv2.resize(result_image, image.shape[:2][::-1])

        # detections, segmentations = model(input_image)
        # segmentations = rescale_segmentation(segmentations, image.shape[0:2])
        # print(f"detections shape: {detections.shape}")
        # print(f"detections shape: {detections.shape}")
        
    # return detections.numpy(), segmentations.cpu().detach().numpy()
    return result_image


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yoeo.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yoeo.pth", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-i", "--images", type=str, default="data/samples", help="Path to directory with images to inference")
    parser.add_argument("-c", "--classes", type=str, default="data/yoeo_names.yaml", help="Path to .yaml file containing the classes' names")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)['detection']  # List of class names

    detect_directory(
        args.model,
        args.weights,
        classes,
        args.output,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres)

if __name__ == '__main__':
    run()