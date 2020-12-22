import glob
import random
import os
import sys
import numpy as np
from PIL import Image

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from .utils import xywh2xyxy_np
import torchvision.transforms as transforms



def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.mask_files = [
            path.replace("images", "masks")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        w, h, _ = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        bounding_boxes = []
        if os.path.exists(label_path):
            boxes = np.loadtxt(label_path).reshape(-1, 5)
            for box in boxes:
                # Extract coordinates for unpadded + unscaled image
                conv_box = xywh2xyxy_np(np.array([box[1:]]))[0]

                conv_box[[0,2]] *= h  
                conv_box[[1,3]] *= w

                bounding_boxes.append(
                    BoundingBox(*conv_box, label=box[0]))

        bounding_boxes = BoundingBoxesOnImage(bounding_boxes, shape=img.shape)

        # ---------
        #  Segmentation Mask
        # ---------

        mask_path = self.mask_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        mask = np.array(Image.open(mask_path).convert('RGB'), dtype=np.uint8) * 255

        segmentation_mask = SegmentationMapsOnImage(mask, shape=img.shape)

        # ---------
        #  Augmentations
        # ---------

        if self.augment or True:
            seq = iaa.Sequential([
                iaa.Dropout([0.0, 0.1]),      # drop 5% or 20% of all pixels
                iaa.Sharpen((0.0, 0.2)),       # sharpen the image
                iaa.Affine(rotate=(-20, 20), translate_percent=(-0.2,0.2)),  # rotate by -45 to 45 degrees (affects segmaps)
                iaa.AddToBrightness((-30, 30)), 
                iaa.AddToHue((-20, 20)),
                iaa.Fliplr(0.5),
                #iaa.ElasticTransformation(alpha=80, sigma=10)  # apply water effect (affects segmaps)
            ], random_order=True)
        else:
            seq = iaa.Sequential([])

        img, bounding_boxes, segmentation_mask = seq(
            image=img, 
            bounding_boxes=bounding_boxes, 
            segmentation_maps=segmentation_mask)

        # ---------
        #  Image
        # ---------

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        bounding_boxes = bounding_boxes.clip_out_of_image()
        #print(bounding_boxes)

        bb_targets = None
        if os.path.exists(label_path):
            boxes = np.zeros((len(bounding_boxes), 5))
            for box_idx, box in enumerate(bounding_boxes):
                # Extract coordinates for unpadded + unscaled image
                x1 = box.x1
                y1 = box.y1
                x2 = box.x2
                y2 = box.y2

                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]

                # Returns (x, y, w, h)
                boxes[box_idx, 0] = box.label
                boxes[box_idx, 1] = ((x1 + x2) / 2) / padded_w
                boxes[box_idx, 2] = ((y1 + y2) / 2) / padded_h
                boxes[box_idx, 3] = (x2 - x1) / padded_h
                boxes[box_idx, 4] = (y2 - y1) / padded_w

            bb_targets = torch.zeros((len(boxes), 6))
            bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        # ---------
        #  Segmentation Mask
        # ---------

        mask = transforms.ToTensor()(segmentation_mask.get_arr())

        # Pad to square resolution
        mask_targets, pad = pad_to_square(mask, 0)

        return img_path, img, bb_targets, mask_targets

    def collate_fn(self, batch):
        paths, imgs, bb_targets, mask_targets= list(zip(*batch))
        # Remove empty placeholder targets
        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        # Stack masks
        mask_targets = torch.stack([resize(mask, self.img_size) for mask in mask_targets])

        return paths, imgs, bb_targets, mask_targets[:,0,:,:].reshape(-1, 1, self.img_size, self.img_size).long()

    def __len__(self):
        return len(self.img_files)
