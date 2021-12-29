#!/usr/bin/env python3

import os
import glob
import yaml
import cv2
import random
import argparse
import numpy as np


##########
# NOTICE #
##########

# This is based on the original 'createYoeoLabels'-script, but adopted to work with the TORSO-21 dataset. https://github.com/bit-bots/TORSO_21_dataset


# Available classes for YOEO
CLASSES = {
    'bb_classes': ['ball', 'goalpost', 'robot'],
    'segmentation_classes': ['field edge', 'lines'],
    'ignored_classes': ['obstacle', 'L-Intersection', 'X-Intersection', 'T-Intersection']
    }


"""
This script reads annotations in the expected yaml format below
to generate the corresponding yolo .txt files and the segmentation masks.


Expected YAML format:
=====================

We use the ``Bit-Bots/YOEOyaml`` export format of the `Bit-Bots ImageTagger <https://imagetagger.bit-bots.de>`_.

set: <imageset_name>
images:
    <image_name>:
        width: <image_width>
        height: <image_height>
        annotations:
          - {
                type: <annotation_type>,
                in_image: true,  # <bool, type is in image>,
                blurred: <bool, is annotation blurred?>,
                concealed: <bool, is annotation concealed?>,
                vector:
                    [
                        [x1,y1],
                        [x2,y2],
                    ]}
          - {
                type: <annotation_type>,
                in_image: false,  # <bool, type is in image>,
    <next_image_name>:
        ...


Expects following file tree:
============================

<superset>/
    - <dataset1>/
        - images/<image_files>
        - <dataset1_annotation_file>.yaml
    - <dataset2>/
        - images/<image_files>
        - <dataset2_annotation_file>.yaml
...


Produces the following file tree:
=================================

<superset>/
    - train.txt
    - test.txt
    - yoeo.names
    - yoeo.data
    - <dataset1>/
        - images/<image_files>
        - labels/<yolo_txt_files>
        - segmentations/<segmentation_mask_files>
        - <dataset1_annotation_file>.yaml
    - <dataset2>/
        - images/<image_files>
        - labels/<yolo_txt_files>
        - segmentations/<segmentation_mask_files>
        - <dataset2_annotation_file>.yaml
...

with train.txt and test.txt containing absolute imagepaths for training and evaluation respectively
with yoeo.names containing the class names of bounding boxes
with yoeo.data: containing number of bounding box classes as well as absolute path to train.txt, test.txt and yoeo.names
"""


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


parser = argparse.ArgumentParser(description="Create YOEO labels from yaml files.")
parser.add_argument("superset", type=str, help="The directory that contains the datasets as subdirectories")
parser.add_argument("testsplit", type=range_limited_float_type_0_to_1, help="Amount of test images from total images: train/test split (between 0.0 and 1.0)")
parser.add_argument("-s", "--seed", type=int, default=random.randint(0, (2**64)-1), help="Seed that controlles the train/test split (integer)")
parser.add_argument("--ignore-blurred", action="store_true", help="Ignore blurred labels")
parser.add_argument("--ignore-conceiled", action="store_true", help="Ignore conceiled labels")
parser.add_argument("--ignore-classes", nargs="+", default=[], help="Append class names, to be ignored")
args = parser.parse_args()

# Remove ignored classes from CLASSES list
for ignore_class in args.ignore_classes:
    for category in CLASSES.keys():
        if ignore_class in CLASSES[category]:
            CLASSES[category].remove(ignore_class)
            print(f"Ignoring class '{ignore_class}'")

imagetagger_annotation_files = glob.glob(f"{args.superset}/*.yaml")
datasets = list(map(lambda x: os.path.basename(os.path.dirname(x)), imagetagger_annotation_files))

datasets_serialized = '\n'.join(datasets)
print(f"The following datasets will be considered: \n{datasets_serialized}")

# Collect image paths for train/test split
images = []

# Iterate over all datasets
for yamlfile in imagetagger_annotation_files:
    d = os.path.dirname(yamlfile)

    print(f"Creating labels for {os.path.basename(d)}")

    labels_dir = os.path.join(d, "labels")
    if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

    masks_dir = os.path.join(d, "yoeo_masks")
    if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

    with open(yamlfile) as f:
        export = yaml.safe_load(f)

    for img_name, frame in export['images'].items():
        # Generate segmentations in correct format
        seg_path = os.path.join(os.path.dirname(yamlfile), "segmentations", os.path.splitext(img_name)[0] + ".png")
        seg_in = cv2.imread(seg_path)
        if seg_in is not None:
            mask = np.zeros(seg_in.shape[:2], dtype=np.uint8)
            mask += ((seg_in == (127, 127, 127)).all(axis=2)).astype(np.uint8)  # Lines
            mask += (((seg_in == (254, 254, 254)).all(axis=2)).astype(np.uint8) * 2)  # Field
            seg_out = np.zeros(seg_in.shape, dtype=np.uint8)
            seg_out[..., 0] = mask
            seg_out[..., 1] = mask
            seg_out[..., 2] = mask
            cv2.imwrite(os.path.join(masks_dir, os.path.splitext(img_name)[0] + ".png"), seg_out)
        else:
            print(f"No segmentation found: '{seg_path}'")
            continue

        images.append(os.path.join(d, "images", img_name))
        name = os.path.splitext(img_name)[0]  # Remove file extension
        imgwidth = frame['width']
        imgheight = frame['height']
        annotations = []

        for annotation in frame['annotations']:
            # Ignore if blurred or conceiled and should be ignored
            if not ((args.ignore_blurred and annotation['blurred']) or
                (args.ignore_conceiled and annotation['conceiled'])):

                if annotation['type'] in CLASSES['bb_classes']:  # Handle bounding boxes
                    if annotation['in_image']:
                        min_x = min(map(lambda x: x[0], annotation['vector']))
                        max_x = max(map(lambda x: x[0], annotation['vector']))
                        min_y = min(map(lambda x: x[1], annotation['vector']))
                        max_y = max(map(lambda x: x[1], annotation['vector']))

                        annowidth = max_x - min_x
                        annoheight = max_y - min_y
                        relannowidth = annowidth / imgwidth
                        relannoheight = annoheight / imgheight

                        center_x = min_x + (annowidth / 2)
                        center_y = min_y + (annoheight / 2)
                        relcenter_x = center_x / imgwidth
                        relcenter_y = center_y / imgheight

                        classID = CLASSES['bb_classes'].index(annotation['type'])  # Derive classID from index in predefined classes
                        annotations.append(f"{classID} {relcenter_x} {relcenter_y} {relannowidth} {relannoheight}")  # Append to store it later
                    else:  # Annotation is not in image
                        continue
                elif annotation['type'] in CLASSES['segmentation_classes']:  # Handle segmentations
                    continue
                    #     if annotation['type'] == 'field edge':
                    #     mask = np.zeros((imgheight, imgwidth, 3), dtype=np.uint8)  # Init black mask

                    #     if not annotation['vector'][0] == 'notinimage':  # Only draw polygon if annotation is known
                    #         vector = [list(pts) for pts in list(annotation['vector'])]
                    #         # Extending the points with corners to fill the space below the horizon
                    #         vector.append([imgwidth - 1, imgheight - 0])
                    #         vector.append([0, imgheight - 1])

                    #         # Generate polygon from vector points
                    #         points = np.array(vector, dtype=np.int32)
                    #         points = points.reshape((1, -1, 2))
                    #         cv2.fillPoly(mask, points, (1, 1, 1))  # Fill mask with not so black polygon

                    #     cv2.imwrite(os.path.join(masks_dir, name + ".png"), mask)  # Store mask on disk
                    # else:
                    #     print(f"The annotation type '{annotation['type']}' is not supported or should be ignored. Image: '{img_name}'")
                elif annotation['type'] in CLASSES['ignored_classes']:  # Ignore this annotation
                    continue
                else:
                    print(f"The annotation type '{annotation['type']}' is not supported or should be ignored. Image: '{img_name}'")

        # Store BB annotations in .txt file
        with open(os.path.join(labels_dir, name + ".txt"), "w") as output:
            output.writelines([annotation + "\n" for annotation in annotations])

# Seed is used for train/test split
random.seed(args.seed)
print(f"Using seed: {args.seed}")

# Generate train/testsplit of images
random.shuffle(sorted(images))  # Sort for consistant order then shuffle with seed
train_images = images[0:round(len(images) * (1 - args.testsplit))]  # Split first range
test_images = images[round(len(images) * (1 - args.testsplit)) + 1:-1]  # Split last range

# Generate meta files
train_images = set(train_images)  # Prevent images from showing up twice
train_path = os.path.join(args.superset, "train.txt")
with open(train_path, "w") as train_file:
    train_file.writelines([image_path + "\n" for image_path in train_images])

test_images = set(test_images)  # Prevent images from showing up twice
test_path = os.path.join(args.superset, "test.txt")
with open(test_path, "w") as test_file:
    test_file.writelines([image_path + "\n" for image_path in test_images])

names_path = os.path.join(args.superset, "yoeo.names")
with open(names_path, "w") as names_file:
    names_file.writelines([class_name + "\n" for class_name in CLASSES['bb_classes']])

data_path = os.path.join(args.superset, "yoeo.data")
with open(data_path, "w") as data_file:
    data_file.write(f"train={train_path}\n")
    data_file.write(f"valid={test_path}\n")
    data_file.write(f"names={names_path}\n")
