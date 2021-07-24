from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from pathlib import Path
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))
        segmaps = np.zeros_like(img)

        # Apply transforms
        if self.transform:
            img, _, _ = self.transform((img, boxes, segmaps))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, data_path, img_size=416, multiscale=True, transform=None):

        # Get all color images for e.g. the test set
        self.img_files = list(Path(data_path).rglob("*.png"))

        """
        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = image_dir.replace("leftImg8bit", "gtFine")
            # TODO bbox stuff
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)
        """

        self.mask_files = []
        for path in self.img_files:
            path = str(path).replace("leftImg8bit", "gtFine")
            mask_dir = os.path.dirname(path)
            mask_file = os.path.join(mask_dir, os.path.basename(path))
            mask_file = os.path.splitext(mask_file)[0] + '_labelIds.png'
            self.mask_files.append(mask_file)

        print(self.mask_files)

        exit(0)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # ---------
        #  Segmentation Mask
        # ---------
        try:
            mask_path = self.mask_files[index % len(self.img_files)].rstrip()
            # Load segmentation mask as numpy array
            mask = np.array(Image.open(mask_path).convert('RGB')) // 127
        except FileNotFoundError as e:
            print(f"Could not load mask '{mask_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets, mask_targets = self.transform(
                    (img, boxes, mask)
                )
            except Exception as e:
                print(f"Could not apply transform.")
                raise e
                return

        return img_path, img, bb_targets, mask_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets, mask_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        # Stack masks and drop the 2 duplicated channels
        mask_targets = torch.stack([resize(mask, self.img_size)[0] for mask in mask_targets]).long()

        return paths, imgs, bb_targets, mask_targets

    def __len__(self):
        return len(self.img_files)
