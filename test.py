from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import cv2

from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, bb_targets, mask_targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += bb_targets[:, 1].tolist()
        # Rescale target
        bb_targets[:, 2:] = xywh2xyxy(bb_targets[:, 2:])
        bb_targets[:, 2:] *= img_size

        imgs_tensor = Variable(imgs.type(Tensor), requires_grad=False)


        with torch.no_grad():
            bb_outputs, segmentation_outputs = model(imgs_tensor)
            bb_outputs = non_max_suppression(bb_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        
        # Pull to cpu, convert to numpy and transpose to usable shape
        imgs = (Variable(imgs.type(Tensor).to("cpu"), requires_grad=False).numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        segs = Variable(segmentation_outputs[0].to("cpu"), requires_grad=False).numpy()[:,:,:,np.newaxis].astype(np.uint8)

        # Iterate over images in batch
        for image_idx_in_batch, img in enumerate(imgs):
            if bb_outputs[image_idx_in_batch] is not None:
                bbs = []
                for box in bb_outputs[image_idx_in_batch]:
                    bbs.append(BoundingBox(
                            x1=box[0], 
                            y1=box[1], 
                            x2=box[2], 
                            y2=box[3],
                            label=["Ball", "Goalpost"][int(box[6])],
                            ))
            else:
                bbs = []

            bbs_in_img = BoundingBoxesOnImage(bbs, shape=img.shape)

            segmap = SegmentationMapsOnImage(segs[image_idx_in_batch], shape=img.shape)

            cv2.imshow("test", bbs_in_img.draw_on_image(segmap.draw_on_image(img)[0]))
            cv2.waitKey(1)

            time.sleep(0.1)

        sample_metrics += get_batch_statistics(bb_outputs, bb_targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
