#!/usr/bin/env python3

import argparse
import json
import os
import random

import cv2
import numpy as np
from pycocotools import mask as coco_mask
from tqdm import tqdm
import yaml

from yoeo.utils.parse_config import parse_data_config


def range_limited_float_type_0_to_1(arg):
    """Type function for argparse - a float within some predefined bounds
    Derived from 'https://stackoverflow.com/questions/55324449/how-to-specify-a-minimum-or-maximum-float-value-with-argparse/55410582#55410582'.
    """
    minimum = 0.0
    maximum = 1.0
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < minimum or f > maximum:
        raise argparse.ArgumentTypeError(f"Argument must be between {minimum} and {maximum}")
    return f


def class_map_type(arg):
    """Type function for argparse - parses an 'OLD=NEW' class name mapping entry."""
    old_name, separator, new_name = arg.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"Invalid class mapping '{arg}', expected format 'OLD_NAME=NEW_NAME'")
    return old_name, new_name


def resolve_label_path(image_path):
    image_dir = os.path.dirname(image_path)
    label_dir = "labels".join(image_dir.rsplit("images", 1))
    return os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")


def resolve_mask_path(image_path):
    image_dir = os.path.dirname(image_path)
    mask_dir = "yoeo_segmentations".join(image_dir.rsplit("images", 1))
    return os.path.join(mask_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")


parser = argparse.ArgumentParser(description="Create YOEO labels from a COCO 1.0 style dataset.")
parser.add_argument("dataset_dir", type=str,
                     help="Path to the dataset directory, expected to contain an 'images' directory and "
                          "'annotations/instances_default.json'. This directory is only ever read from, never written to.")
parser.add_argument("destination_dir", type=str,
                     help="Directory the YOEO formatted dataset (train/test split, labels, segmentation masks, "
                          "symlinked images) will be written to.")
parser.add_argument("--annotations-file", type=str, default=None,
                     help="Path to the COCO annotations json file. Defaults to '<dataset_dir>/annotations/instances_default.json'.")
parser.add_argument("--images-dir", type=str, default=None,
                     help="Path to the directory containing the dataset images. Defaults to '<dataset_dir>/images'.")
parser.add_argument("--test-split", type=range_limited_float_type_0_to_1, default=0.2,
                     help="Fraction of images to put into the test partition.")
parser.add_argument("--seed", type=int, default=42,
                     help="Seed used to deterministically shuffle the images before splitting them into the train/test partitions.")
parser.add_argument("--extend-dataset", type=str, default=None,
                     help="Path to a '.data' file of an existing YOEO formatted dataset (e.g. TORSO-21) to merge into "
                          "the output, so the result can be trained on as one combined dataset. Its existing train/test "
                          "split is preserved. The extended dataset is only ever read from, never modified.")
parser.add_argument("--class-map", type=class_map_type, nargs="+", default=[],
                     help="Explicit 'OLD_NAME=NEW_NAME' class name overrides applied to --extend-dataset's class names "
                          "before checking them against this dataset's classes, e.g. 'pitch=field' if the two datasets "
                          "name the same class differently.")
parser.add_argument("--allow-new-extend-classes", action="store_true",
                     help="If --extend-dataset has classes that don't exist in this dataset (and aren't covered by "
                          "--class-map), add them as new classes instead of raising an error.")
args = parser.parse_args()

# Available classes for YOEO.
# We currently only care about the ball and other robots for detection, and lines/field/background for segmentation.
CLASSES = {
    'bb_classes': ['ball', 'robot'],
    'segmentation_classes': ['background', 'lines', 'field'],
}

# Maps COCO category names to YOEO detection classes
BB_CATEGORY_NAME_MAP = {
    'soccer ball': 'ball',
    'robot': 'robot',
}

# Maps COCO category names to YOEO segmentation classes.
# Painted in this order, i.e. later entries are painted on top of earlier ones (z-index).
# Background is implicit (it is simply never painted over).
SEGMENTATION_PAINT_ORDER = ['field', 'lines']
SEGMENTATION_CATEGORY_NAME_MAP = {
    'football field': 'field',
    'field lines': 'lines',
}

dataset_dir = args.dataset_dir
assert os.path.exists(dataset_dir), f"Is the given path correct? Directory does not exist: '{dataset_dir}'"

annotations_file = args.annotations_file or os.path.join(dataset_dir, "annotations", "instances_default.json")
images_dir = args.images_dir or os.path.join(dataset_dir, "images")
assert os.path.exists(annotations_file), f"Is the given path correct? File does not exist: '{annotations_file}'"
assert os.path.exists(images_dir), f"Is the given path correct? Directory does not exist: '{images_dir}'"

destination_dir = args.destination_dir
os.makedirs(destination_dir, exist_ok=True)

print(f"Loading annotation file: '{annotations_file}'...")
with open(annotations_file, 'r') as f:
    data = json.load(f)

# Map COCO category id -> COCO category name
category_id_to_name = {category['id']: category['name'] for category in data['categories']}

# Warn once about categories that are neither used for detection nor for segmentation
unsupported_categories = {
    name for name in category_id_to_name.values()
    if name not in BB_CATEGORY_NAME_MAP and name not in SEGMENTATION_CATEGORY_NAME_MAP
}
for name in unsupported_categories:
    print(f"Ignoring unsupported category '{name}'")

# Group annotations by image id
annotations_by_image_id = {}
for annotation in data['annotations']:
    annotations_by_image_id.setdefault(annotation['image_id'], []).append(annotation)

images_by_id = {image['id']: image for image in data['images']}

# Deterministically shuffle and split the images into a train and a test partition
image_ids = sorted(images_by_id.keys())
random.Random(args.seed).shuffle(image_ids)
num_test_images = round(len(image_ids) * args.test_split)
partitions = {
    'test': image_ids[:num_test_images],
    'train': image_ids[num_test_images:],
}

for partition, partition_image_ids in partitions.items():
    # Paths in destination dir
    partition_destination_dir = os.path.join(destination_dir, partition)
    partition_images_dir = os.path.join(partition_destination_dir, "images")
    labels_dir = os.path.join(partition_destination_dir, "labels")
    yoeo_segmentation_dir = os.path.join(partition_destination_dir, "yoeo_segmentations")

    os.makedirs(partition_images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(yoeo_segmentation_dir, exist_ok=True)

    image_names = []

    print(f"Writing '{partition}' partition ({len(partition_image_ids)} images)...")
    for image_id in tqdm(partition_image_ids):
        image_data = images_by_id[image_id]
        img_name_with_extension = image_data['file_name']
        img_name_without_extension = os.path.splitext(img_name_with_extension)[0]
        img_width = image_data['width']
        img_height = image_data['height']

        image_names.append(img_name_with_extension)

        # SEGMENTATIONS
        ###############

        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        annotations_by_segmentation_class = {}
        for annotation in annotations_by_image_id.get(image_id, []):
            category_name = category_id_to_name[annotation['category_id']]
            segmentation_class = SEGMENTATION_CATEGORY_NAME_MAP.get(category_name)
            if segmentation_class is not None:
                annotations_by_segmentation_class.setdefault(segmentation_class, []).append(annotation)

        # Paint in z-index order, so that e.g. lines end up on top of the field
        for segmentation_class in SEGMENTATION_PAINT_ORDER:
            class_id = CLASSES['segmentation_classes'].index(segmentation_class)
            for annotation in annotations_by_segmentation_class.get(segmentation_class, []):
                binary_mask = coco_mask.decode(annotation['segmentation']).astype(bool)
                mask[binary_mask] = class_id

        seg_out = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        cv2.imwrite(os.path.join(yoeo_segmentation_dir, img_name_without_extension + ".png"), seg_out)

        # BOUNDING BOXES
        ################

        bb_annotations = []
        for annotation in annotations_by_image_id.get(image_id, []):
            category_name = category_id_to_name[annotation['category_id']]
            bb_class = BB_CATEGORY_NAME_MAP.get(category_name)
            if bb_class is None:
                continue

            x, y, box_width, box_height = annotation['bbox']

            # Clip the box to the image bounds, as some annotations slightly overflow the image
            min_x = max(0.0, x)
            min_y = max(0.0, y)
            max_x = min(float(img_width), x + box_width)
            max_y = min(float(img_height), y + box_height)

            annotation_width = max_x - min_x
            annotation_height = max_y - min_y
            if annotation_width <= 0 or annotation_height <= 0:
                continue

            relative_annotation_width = annotation_width / img_width
            relative_annotation_height = annotation_height / img_height
            relative_center_x = (min_x + annotation_width / 2) / img_width
            relative_center_y = (min_y + annotation_height / 2) / img_height

            class_id = CLASSES['bb_classes'].index(bb_class)
            bb_annotations.append(f"{class_id} {relative_center_x} {relative_center_y} {relative_annotation_width} {relative_annotation_height}")

        with open(os.path.join(labels_dir, img_name_without_extension + ".txt"), "w") as output:
            output.writelines([annotation + "\n" for annotation in bb_annotations])

        # Symlink the image into the destination directory, since the source dataset must not be modified
        link_path = os.path.join(partition_images_dir, img_name_with_extension)
        target_path = os.path.join(images_dir, img_name_with_extension)
        if not os.path.exists(link_path):
            os.symlink(target_path, link_path)

    # Write train.txt or test.txt file containing full paths to each image
    partition_txt_path = os.path.join(destination_dir, partition, f"{partition}.txt")
    with open(partition_txt_path, "w") as partition_txt_file:
        partition_txt_file.writelines([str(os.path.join(partition_images_dir, image_name)) + "\n" for image_name in image_names])

# EXTEND WITH ANOTHER DATASET
##############################

if args.extend_dataset:
    class_map = dict(args.class_map)

    print(f"Extending dataset with '{args.extend_dataset}'...")
    extend_options = parse_data_config(args.extend_dataset)
    with open(extend_options['names'], 'r') as f:
        extend_names = yaml.safe_load(f)

    def resolve_class_index_map(kind, extend_class_list, our_class_list):
        """Maps each class id in 'extend_class_list' onto its id in 'our_class_list' (by name, after applying
        --class-map overrides), appending genuinely new classes to 'our_class_list' if allowed."""
        index_map = []
        for class_name in extend_class_list:
            mapped_name = class_map.get(class_name, class_name)
            if mapped_name not in our_class_list:
                if not args.allow_new_extend_classes:
                    raise ValueError(
                        f"--extend-dataset has {kind} class '{class_name}' (mapped to '{mapped_name}') which is not "
                        f"one of this dataset's {kind} classes {our_class_list}. Use --class-map OLD=NEW to map it "
                        "onto an existing class, or pass --allow-new-extend-classes to add it as a new class."
                    )
                print(f"Adding new {kind} class '{mapped_name}' from --extend-dataset")
                our_class_list.append(mapped_name)
            index_map.append(our_class_list.index(mapped_name))
        return index_map

    detection_index_map = resolve_class_index_map('detection', extend_names['detection'], CLASSES['bb_classes'])
    segmentation_index_map = resolve_class_index_map('segmentation', extend_names['segmentation'], CLASSES['segmentation_classes'])

    extend_tag = os.path.splitext(os.path.basename(args.extend_dataset))[0]

    for partition, list_key in (('train', 'train'), ('test', 'valid')):
        partition_destination_dir = os.path.join(destination_dir, partition)
        partition_images_dir = os.path.join(partition_destination_dir, "images")
        labels_dir = os.path.join(partition_destination_dir, "labels")
        yoeo_segmentation_dir = os.path.join(partition_destination_dir, "yoeo_segmentations")
        os.makedirs(partition_images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(yoeo_segmentation_dir, exist_ok=True)

        with open(extend_options[list_key], 'r') as f:
            extend_image_paths = [line.strip() for line in f if line.strip()]

        extend_image_names = []
        print(f"Merging {len(extend_image_paths)} '{partition}' images from '{extend_tag}'...")
        for image_path in tqdm(extend_image_paths):
            joined_name = f"{extend_tag}__{os.path.basename(image_path)}"
            joined_name_without_extension = os.path.splitext(joined_name)[0]
            extend_image_names.append(joined_name)

            # Remap and copy the bounding box labels
            source_label_path = resolve_label_path(image_path)
            remapped_lines = []
            if os.path.exists(source_label_path):
                with open(source_label_path, 'r') as label_file:
                    for line in label_file:
                        line = line.strip()
                        if not line:
                            continue
                        class_id, *rest = line.split()
                        remapped_class_id = detection_index_map[int(class_id)]
                        remapped_lines.append(f"{remapped_class_id} {' '.join(rest)}")
            else:
                print(f"No label file found: '{source_label_path}'")
            with open(os.path.join(labels_dir, joined_name_without_extension + ".txt"), "w") as label_out:
                label_out.writelines([line + "\n" for line in remapped_lines])

            # Remap and write the segmentation mask
            source_mask_path = resolve_mask_path(image_path)
            mask_in = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_in is not None:
                lut = np.arange(256, dtype=np.uint8)
                lut[:len(segmentation_index_map)] = segmentation_index_map
                remapped_mask = lut[mask_in]
                seg_out = np.repeat(remapped_mask[:, :, np.newaxis], 3, axis=2)
                cv2.imwrite(os.path.join(yoeo_segmentation_dir, joined_name_without_extension + ".png"), seg_out)
            else:
                print(f"No segmentation mask found: '{source_mask_path}'")

            # Symlink the image into the destination directory, since the extended dataset must not be modified
            link_path = os.path.join(partition_images_dir, joined_name)
            target_path = os.path.abspath(image_path)
            if not os.path.exists(link_path):
                os.symlink(target_path, link_path)

        partition_txt_path = os.path.join(destination_dir, partition, f"{partition}.txt")
        with open(partition_txt_path, "a") as partition_txt_file:
            partition_txt_file.writelines(
                [str(os.path.join(partition_images_dir, image_name)) + "\n" for image_name in extend_image_names])

# The names file contains the class names of bb detections and segmentations
names_path = os.path.join(destination_dir, "yoeo_names.yaml")
names = {
    'detection': CLASSES['bb_classes'],
    'segmentation': CLASSES['segmentation_classes'],
}
with open(names_path, "w") as names_file:
    yaml.dump(names, names_file)

data_path = os.path.join(destination_dir, "yoeo.data")
with open(data_path, "w") as data_file:
    data_file.write(f"train={os.path.join(destination_dir, 'train', 'train.txt')}\n")
    data_file.write(f"valid={os.path.join(destination_dir, 'test', 'test.txt')}\n")
    data_file.write(f"names={names_path}\n")
