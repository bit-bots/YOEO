#! /usr/bin/env python3

from __future__ import division

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from yoeo.models import load_model
from yoeo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info, seg_iou
from yoeo.utils.datasets import ListDataset
from yoeo.utils.transforms import DEFAULT_TRANSFORMS
from yoeo.utils.parse_config import parse_data_config

import cv2

def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: Dict containing detection and segmentation class names
    :type class_names: Dict
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output, seg_class_ious = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output, seg_class_ious


def print_eval_stats(metrics_output, seg_class_ious, class_names, verbose):
    # Print detection statistics
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names['detection'][c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


    # Print segmentation statistics
    if verbose:
        # Print IoU per segmentation class
        seg_table = [["Index", "Class", "IoU"]]
        for i, iou in enumerate(seg_class_ious):
            seg_table += [[i, class_names['segmentation'][i], "%.5f" % iou]]
        print(AsciiTable(seg_table).table)
    # Print mean IoU
    mean_seg_class_ious = np.array(seg_class_ious).mean()
    print(f"----Average IoU {mean_seg_class_ious:.5f} ----")


def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: Dict containing detection and segmentation class names
    :type class_names: Dict
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    seg_ious = []
    import time
    times=[]
    for _, imgs, bb_targets, mask_targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += bb_targets[:, 1].tolist()
        # Rescale target
        bb_targets[:, 2:] = xywh2xyxy(bb_targets[:, 2:])
        bb_targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            t1 = time.time()
            yolo_outputs, segmentation_outputs = model(imgs)
            times.append(time.time() - t1)
            yolo_outputs = non_max_suppression(yolo_outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(yolo_outputs, bb_targets, iou_threshold=iou_thres)

        """
        print("Targets", np.unique(to_cpu(mask_targets), return_counts=True))
        mask = to_cpu(mask_targets[0].squeeze(0)) * 127
        cv2.imshow("Seg", np.array(mask.numpy(), dtype=np.uint8))
        cv2.waitKey(0)
        """

        seg_ious.append(seg_iou(to_cpu(segmentation_outputs), mask_targets, model.num_seg_classes))

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    print(f"Times: Mean {1/np.array(times).mean()}fps | Std: {np.array(times).std()} ms")

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    yolo_metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    seg_class_ious = [np.array(class_ious).mean() for class_ious in list(zip(*seg_ious))]

    print_eval_stats(yolo_metrics_output, seg_class_ious, class_names, verbose)

    return yolo_metrics_output, seg_class_ious


def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

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
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yoeo.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yoeo.pth", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/torso.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # Detection and segmentation class names

    evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)


if __name__ == "__main__":
    run()
