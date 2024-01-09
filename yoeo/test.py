#! /usr/bin/env python3

from __future__ import division, annotations
from typing import List, Optional, Tuple

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from yoeo.models import load_model
from yoeo.utils.utils import ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, \
    print_environment_info, seg_iou
from yoeo.utils.datasets import ListDataset
from yoeo.utils.transforms import DEFAULT_TRANSFORMS
from yoeo.utils.dataclasses import ClassNames
from yoeo.utils.class_config import ClassConfig
from yoeo.utils.parse_config import parse_data_config
from yoeo.utils.metric import Metric


def evaluate_model_file(model_path, weights_path, img_path, class_config, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_config: Object containing all class name related settings
    :type class_config: TrainConfig
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
    metrics_output, seg_class_ious, secondary_metric = _evaluate(
        model,
        dataloader,
        class_config,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output, seg_class_ious, secondary_metric


def print_eval_stats(metrics_output: Optional[Tuple[np.ndarray]], 
                     seg_class_ious: List[np.float64], 
                     secondary_metric: Optional[Metric], 
                     class_config: ClassConfig, 
                     verbose: bool
                     ):
    # Print detection statistics
    print("#### Detection ####")
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            class_names = class_config.get_squeezed_det_class_names()
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

    if secondary_metric is not None:
        print("#### Detection - Secondary ####")
        mbACC = secondary_metric.mbACC()

        if verbose:
            classes = class_config.get_group_class_names()
            mbACC_per_class = [secondary_metric.bACC(i) for i in range(len(classes))]
                        
            sec_table = [["Index", "Class", "bACC"]]
            for i, c in enumerate(classes):
                sec_table += [[i, c, "%.5f" % mbACC_per_class[i]]]
            print(AsciiTable(sec_table).table)

        print(f"---- mbACC {mbACC:.5f} ----")

    print("#### Segmentation ####")
    # Print segmentation statistics
    if verbose:
        # Print IoU per segmentation class
        seg_table = [["Index", "Class", "IoU"]]
        class_names = class_config.get_seg_class_names()
        for i, iou in enumerate(seg_class_ious):
            seg_table += [[i, class_names[i], "%.5f" % iou]]
        print(AsciiTable(seg_table).table)
    # Print mean IoU
    mean_seg_class_ious = np.array(seg_class_ious).mean()
    print(f"----Average IoU {mean_seg_class_ious:.5f} ----")


def _evaluate(model, dataloader, class_config, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_config: Object storing all class related settings
    :type class_config: TrainConfig
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
    times = []

    if class_config.classes_should_be_grouped():
        secondary_metric = Metric(len(class_config.get_group_ids()))
    else:
        secondary_metric = None

    for _, imgs, bb_targets, mask_targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += bb_targets[:, 1].tolist()

        # If a subset of the detection classes should be grouped into one class for non-maximum suppression and the 
        # subsequent AP-computation, we need to group those class labels here.
        if class_config.classes_should_be_grouped():
            labels = class_config.group(labels)

        # Rescale target
        bb_targets[:, 2:] = xywh2xyxy(bb_targets[:, 2:])
        bb_targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            t1 = time.time()
            yolo_outputs, segmentation_outputs = model(imgs)
            times.append(time.time() - t1)
            yolo_outputs = non_max_suppression(
                yolo_outputs,
                conf_thres=conf_thres,
                iou_thres=nms_thres,
                group_config=class_config.get_group_config()
            )

        sample_stat, secondary_stat = get_batch_statistics(
            yolo_outputs, 
            bb_targets, 
            iou_threshold=iou_thres, 
            group_config=class_config.get_group_config()
        )

        sample_metrics += sample_stat

        if class_config.classes_should_be_grouped():
            secondary_metric += secondary_stat

        seg_ious.append(seg_iou(to_cpu(segmentation_outputs), mask_targets, model.num_seg_classes))

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    print(f"Times: Mean {1 / np.array(times).mean()}fps | Std: {np.array(times).std()} ms")

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    
    yolo_metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    def seg_iou_mean_without_nan(seg_iou: List[float]) -> np.ndarray:
        """This helper function is needed to remove cases, where the segmentation IOU is NaN.
        This is the case, if a whole batch does not contain any pixels of a segmentation class.

        :param seg_iou: Segmentation IOUs, possibly including NaN
        :return: Segmentation IOUs without NaN
        """
        seg_iou = np.asarray(seg_iou)
        return seg_iou[~np.isnan(seg_iou)].mean()

    seg_class_ious = [seg_iou_mean_without_nan(class_ious) for class_ious in list(zip(*seg_ious))]

    print_eval_stats(yolo_metrics_output, seg_class_ious, secondary_metric, class_config, verbose)

    return yolo_metrics_output, seg_class_ious, secondary_metric


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
    parser.add_argument("-m", "--model", type=str, default="config/yoeo.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yoeo.pth",
                        help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/torso.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    parser.add_argument("--class_config", type=str, default="class_config/default.yaml", help="Class configuration for evaluation")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]

    class_names = ClassNames.load_from(data_config["names"])  # Detection and segmentation class names
    class_config = ClassConfig.load_from(args.class_config, class_names)

    evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_config,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    run()
