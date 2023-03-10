#! /usr/bin/env python3
from __future__ import division

import os
import argparse
import tqdm
import numpy as np
import cv2
import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from yoeo.models import load_model
from yoeo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info, rescale_segmentation
from yoeo.utils.datasets import ImageFolder
from yoeo.utils.transforms import Resize, DEFAULT_TRANSFORMS

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
    model = load_model(model_path, weights_path)
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

        detections, segmentations = detect_image(
            model,
            image,
            img_size,
            conf_thres,
            nms_thres)
        print("NY PRIVET last")

        _draw_and_save_output_image(image, detections, segmentations, img_size, output_path, classes)
        print(time.time() - t)
        print("END OF INFERENCE OF IMAGE")

    # print(f"---- Detections were saved to: '{output_path}' ----")

        print(f"SUM UP: {image.shape}")

        if cv2.waitKey(1) == 27:
            cam.release()
            cv2.destroyAllWindows()
            break


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
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
    model.eval()  # Set model to evaluation mode
    
    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])((
            image,
            np.empty((1, 5)),
            np.empty((img_size, img_size), dtype=np.uint8)))[0].unsqueeze(0)
    
    print(f"raw image shape: {image.shape}")
    print(f"torch image shape: {input_img.shape}")
    print(f"torch image shape: {type(input_img)}")
    

    # if torch.cuda.is_available():
    #     input_img = input_img.to("cuda")
    print(f"model: {model}")
    # Get detections
    with torch.no_grad():
        detections, segmentations = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[0:2])
        segmentations = rescale_segmentation(segmentations, image.shape[0:2])
        print(f"detections shape: {detections.shape}")
        print(f"detections shape: {detections.shape}")
        
    # return detections.numpy(), segmentations.cpu().detach().numpy()
    return detections, segmentations

def detect(model, output_path, conf_thres, nms_thres):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)
    print("nychetiii 1")
    model.eval()  # Set model to evaluation mode
    print("nychetiii 2")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor



    cam = cv2.VideoCapture(0)
    key = cv2.waitKey(1)
    while key != 27:
        _, image = cam.read()
        cv2.imshow('raw', image)

        img = torch.from_numpy(image)
        img = Variable(img.type(Tensor))
        print(img)
        # Get detections
        with torch.no_grad():
            print(img)
            detections, segmentations = model(img)
            print("NY PRIVET 8")
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        if detections and segmentations and img:
            return detections, segmentations, img
    
    if cv2.waitkey(1) == 27:
        cam.release()
        cv2.destroyAllWindows()


def _draw_and_save_output_image(image, detections, seg, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param seg: Segmentation image
    :type seg: Tensor
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.figure()
    fig, ax = plt.subplots(1)
    # Get segmentation
    seg = seg.cpu().detach().numpy().astype(np.uint8)
    # seg = seg.astype(np.uint8)
    # Draw all of it
    seg = seg[0]
    print(f"ETO EST SEG {seg}")
    # The amount of padding that was added
    print("GOVNINA")
    print(img_size / max(img.shape[:2]))
    print(max(img.shape[0] - img.shape[1], 0))
    print(img.shape[0], img.shape[1])
    print("end of GOVNINA")
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape[:2])) // 2
    # pad_x = 21.0
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape[:2])) // 2
    print(f"CHEKAI PADI {pad_x, pad_y}")

    seg_map = seg[
                int(pad_y) : int(img_size - pad_y),
                int(pad_x) : int(img_size - pad_x),
                ] * 255

    print(f"MILLIARDNAYA {img, pad_y, pad_x}")
    ax.imshow(
        SegmentationMapsOnImage(
            seg[
                int(pad_y) : int(img_size - pad_y),
                int(pad_x) : int(img_size - pad_x),
                ], shape=img.shape).draw_on_image(img)[0])
    print("JJEEPPAAA")
    # Rescale boxes to original image

    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=colors[int(cls_pred)], facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        """
        plt.text(
            x1,
            y1,
            s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(cls_pred)], "pad": 0})
        """

    # Save generated image with detections
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    # filename = os.path.basename(image_path).split(".")[0]
    # output_path_1 = os.path.join(output_path, f"{filename}.png")
    # redraw the canvas
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                    sep='')
    print(f"summing up PICTURE 0 : {img.shape}")                                
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    print(f"summing up PICTURE 1 : {img.shape}")
    # cv2.imwrite(output_path_1, img)
    cv2.imshow('inference', img)
    # cv2.waitKey(1)


def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """

    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


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
