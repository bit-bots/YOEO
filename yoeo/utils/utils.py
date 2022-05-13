from __future__ import division

from typing import Tuple

import time
import platform
import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import subprocess
import random
from typing import List
import yaml


def provide_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path: str) -> dict:
    with open(path, 'r', encoding="utf-8") as fp:
        names = yaml.load(fp, Loader=yaml.SafeLoader)

    assert "detection" in names.keys(), f"Missing key 'detection' in {path}"
    assert "segmentation" in names.keys(), f"Missing key 'segmentation' in {path}"

    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, output_img_size, original_img_size):
    """
    Rescale bounding boxes as if they were calculated on the original, non-padded image.
    1. bounding boxes are scaled as if they were calculated on the (square) padded original image.
    2. padding is subtracted, thereby shifting the boxes as if they were calculated on the original,
       non-padded image.

    :param boxes: detection output
    :type boxes: torch.Tensor with shape(#boxes, 6)
    :param output_img_size: size of the image for which the network calculates the bounding boxes (1D)
    :type output_img_size: int
    :param original_img_size: size of the original image (height, width)
    :type original_img_size: Tuple[int, int] (height, width)
    :return: rescaled detection output
    :rtype: torch.Tensor with shape(#boxes, 6)
    """

    rescaled_boxes = rescale_boxes_to_original_padded_img_size(boxes, output_img_size, max(original_img_size))
    rescaled_boxes = unpad_box_coordinates(rescaled_boxes, original_img_size)

    return rescaled_boxes


def rescale_boxes_to_original_padded_img_size(boxes, output_img_size: int, original_max_size: int):
    """
    Rescale bounding boxes as if they were calculated on the (square) padded original image.

    :param boxes: detection output
    :type boxes: torch.Tensor with shape(#boxes, 6)
    :param output_img_size: size of the image for which the network calculates the bounding boxes (1D)
    :type output_img_size: int
    :param original_max_size: maximum size of the original image (1)
    :type original_max_size: int
    :return: rescaled detection output
    :rtype: torch.Tensor with shape(#boxes, 6)
    """

    scale_factor = original_max_size / output_img_size
    boxes[:, 0:4] = boxes[:, 0:4] * scale_factor

    return boxes


def unpad_box_coordinates(boxes, original_img_size: Tuple[int, int]):
    """
    Subtract padding, thereby shifting the boxes as if they were calculated on the original,
    non-padded image.

    :param boxes: detection output
    :type boxes: torch.Tensor with shape(#boxes, 6)
    :param original_img_size: size of the original image (height, width)
    :type original_img_size: Tuple[int, int] (height, width)
    :return: rescaled detection output
    :rtype: torch.Tensor with shape(#boxes, 6)
    """

    padding_left = max(original_img_size[0] - original_img_size[1], 0) // 2
    padding_top = max(original_img_size[1] - original_img_size[0], 0) // 2

    boxes[:, 0] = boxes[:, 0] - padding_left
    boxes[:, 1] = boxes[:, 1] - padding_top
    boxes[:, 2] = boxes[:, 2] - padding_left
    boxes[:, 3] = boxes[:, 3] - padding_top

    return boxes


def rescale_segmentation(segmentation, original_img_size: Tuple[int, int]):
    """
    Interpolate segmentation back to original image size and remove paddings.
    1. segmentation is rescaled as if it was calculated on the original, padded image size
    2. paddings are removed, thereby restoring the original image size

    :param segmentation: segmentation output
    :type segmentation: torch.Tensor with shape (1, height, width) and height == width
    :param original_img_size: size of the original image (height, width)
    :type original_img_size: Tuple[int, int] (height, width)
    :return: rescaled segmentation
    :rtype: torch.Tensor with shape (1, original_img_size[0], original_img_size[1])
    """

    rescaled_seg = rescale_segmentation_to_original_padded_img_size(segmentation, max(original_img_size))
    rescaled_seg = unpad_segmentation(rescaled_seg, original_img_size)

    return rescaled_seg


def rescale_segmentation_to_original_padded_img_size(segmentation, original_max_size: int):
    """
    Rescale the segmentation as if it was calculated on the original, padded image size using
    "nearest-exact" interpolation.

    :param segmentation: segmentation output
    :type segmentation: torch.Tensor with shape (1, height, width) and height == width
    :param original_max_size: maximum size of the original image (1)
    :type original_max_size: int
    :return: segmentation output with original, padded image size
    :rtype: torch.Tensor with shape (1, original_max_size, original_max_size)
    """

    return nn.functional.interpolate(
        segmentation.unsqueeze(0),
        size=(original_max_size, original_max_size),
        mode="nearest-exact"
    ).squeeze(0)


def unpad_segmentation(segmentation, original_img_size: Tuple[int, int]):
    """
    Remove paddings, thereby restoring the original image size
    
    :param segmentation: segmentation output
    :type segmentation: torch.Tensor with shape (1, height, width) and height == width
    :param original_img_size: original image size (height, width)
    :type original_img_size: Tuple[int, int]
    :return: unpadded segmentation output
    :rtype: torch.Tensor with shape (1, original_img_size[0], original_img_size[1])
    """

    current_size = segmentation.size(dim=1)
    original_height, original_width = original_img_size

    total_vertical_padding = max(0, original_width - original_height)
    total_horizontal_padding = max(0, original_height - original_width)

    padding_top = total_vertical_padding // 2
    padding_bottom = total_vertical_padding - padding_top
    padding_left = total_horizontal_padding // 2
    padding_right = total_horizontal_padding - padding_left

    return segmentation[..., padding_top:current_size - padding_bottom, padding_left:current_size - padding_right]


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(
                    *filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def seg_iou(pred, target, classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious


def print_environment_info():
    """
    Prints infos about the environment and the system.
    This should help when people make issues containg the printout.
    """

    print("Environment information:")

    # Print OS information
    print(f"System: {platform.system()} {platform.release()}")

    # Print poetry package version
    try:
        print(
            f"Current Version: {subprocess.check_output(['poetry', 'version'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Not using the poetry package")

    # Print commit hash if possible
    try:
        print(
            f"Current Commit Hash: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No git or repo found")
