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
        # Create augmentation input dictionary
        augmentation_input = {
            "image": data["img"]
        }

        # Check if we have boxes to augment
        if "boxes" in data.keys():
            # Convert xywh to xyxy
            boxes = np.array(data["boxes"])
            boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

            # Convert bounding boxes to imgaug
            augmentation_input["bounding_boxes"] = BoundingBoxesOnImage(
                [BoundingBox(*box[1:], label=box[0]) for box in boxes],
                shape=data["img"].shape)

        # Check if we have segmentations to augment
        if "seg" in data.keys():
            # Convert sementations to imgaug
            augmentation_input["segmentation_mask"] = SegmentationMapsOnImage(data["seg"], shape=data["img"].shape)

        # Apply augmentations
        augmentation_output = self.augmentations(**augmentation_input)

        # Get augmented image
        data["img"] = augmentation_output[0]

        # Check if we have augmented boxes
        if "boxes" in data.keys():
            # Clip out of image boxes
            bounding_boxes = augmentation_output[1].clip_out_of_image()

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

                data["boxes"] = boxes

        # Check if we have augmented segmentations
        if "seg" in data.keys():
            # Convert segmentation back to numpy
            data["seg"] = augmentation_output[2].get_arr()

        return data


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        # Check if there are boxes and an image in the data
        if "boxes" in data.keys() and "img" in data.keys():
            h, w, _ = data["img"].shape
            data["boxes"][:,[1, 3]] /= w
            data["boxes"][:,[2, 4]] /= h
        return data


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        # Check if there are boxes and an image in the data
        if "boxes" in data.keys() and "img" in data.keys():
            h, w, _ = data["img"].shape
            data["boxes"][:,[1, 3]] *= w
            data["boxes"][:,[2, 4]] *= h
        return data


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
        # Check if there is an image in the data
        if "img" in data.keys():
            # Extract image as PyTorch tensor
            data["img"] = transforms.ToTensor()(data["img"])
        # Check if there is an segmentation in the data
        if "seg" in data.keys():
            data["seg"] = transforms.ToTensor()(data["seg"]) * 255 # Because troch maps this to 0-1 instead of 0-255
        # Check if there are boxes in the data
        if "boxes" in data.keys():
            data["boxes"] = torch.zeros((len(data["boxes"]), 6))
            data["boxes"][:, 1:] = transforms.ToTensor()(data["boxes"])
        return data


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        # Check if there is an image in the data
        if "img" in data.keys():
            data["img"] = F.interpolate(data["img"].unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        # Check if there is an segmentation in the data
        if "seg" in data.keys():
            data["seg"]  = F.interpolate(data["seg"] .unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return data


DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])
	