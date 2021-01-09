import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from .utils import xywh2xyxy_np
import torchvision.transforms as transforms


class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        img, boxes, seg = data

        # Convert bounding boxes to imgaug        
        bounding_boxes = []
        for box in boxes:
            # Extract coordinates for unpadded + unscaled image
            conv_box = xywh2xyxy_np(np.array([box[1:]]))[0]

            bounding_boxes.append(
                BoundingBox(*conv_box, label=box[0]))
        bounding_boxes = BoundingBoxesOnImage(bounding_boxes, shape=img.shape)

        # Convert sementations to imgaug
        segmentation_mask = SegmentationMapsOnImage(seg, shape=img.shape)

        # ---------
        #  Augmentations
        # ---------
        img, bounding_boxes, segmentation_mask = self.augmentations(
            image=img, 
            bounding_boxes=bounding_boxes, 
            segmentation_maps=segmentation_mask)


        # Clip out of image
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        # Convert segmentation back to numpy
        seg = segmentation_mask.get_arr()

        return img, boxes, seg


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes, seg = data
        w, h, _ = img.shape 
        boxes[:,[1,3]] /= h
        boxes[:,[2,4]] /= w
        return img, boxes, seg


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes, seg = data
        w, h, _ = img.shape 
        boxes[:,[1,3]] *= h
        boxes[:,[2,4]] *= w
        return img, boxes, seg


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
            ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes, seg = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)
        seg = transforms.ToTensor()(seg)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets, seg


DEFAULT_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])