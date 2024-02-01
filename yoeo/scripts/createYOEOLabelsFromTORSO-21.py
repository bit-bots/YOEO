#!/usr/bin/env python3

import os
import yaml
import cv2
import argparse
import numpy as np
from tqdm import tqdm


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
parser.add_argument("dataset_collection_dir", type=str, help="Directory to a TORSO-21 collection (reality or simulation). Output will be written here, unless --destination-dir is given.")
parser.add_argument("--destination-dir", type=str, default="", help="Writes output files to specified directory. Also creates symlinks form image files to destination-dir. Useful, when using read-only datasets.")
parser.add_argument("--skip-blurred", action="store_true", help="Skip blurred labels")
parser.add_argument("--skip-concealed", action="store_true", help="Skip concealed labels")
parser.add_argument("--skip-classes", nargs="+", default=[], help="These bounding box classes will be skipped")
parser.add_argument("--robots-with-team-colors", action="store_true", help="The robot class will be subdivided into subclasses, one for each team color (currently either 'blue', 'red' or 'unknown').")
args = parser.parse_args()

# Available classes for YOEO
CLASSES = {
    'bb_classes': ['ball', 'goalpost', 'robot'] if not args.robots_with_team_colors else ['ball', 'goalpost', 'robot_blue', 'robot_red', 'robot_unknown'],
    'segmentation_classes': ['background', 'lines', 'field'],
    'skip_classes': ['obstacle', 'L-Intersection', 'X-Intersection', 'T-Intersection'],
    }

# Remove skipped classes from CLASSES list
for skip_class in args.skip_classes:
    if skip_class in CLASSES['bb_classes']:
        CLASSES['bb_classes'].remove(skip_class)
        CLASSES['skip_classes'].append(skip_class)
        print(f"Ignoring class '{skip_class}'")

# Defaults
create_symlinks = False

# Handle dataset paths
assert os.path.exists(args.dataset_collection_dir), f"Is the given path correct? Directory does not exist: '{args.dataset_collection_dir}'"
dataset_collection_dir = args.dataset_collection_dir
destination_dir = args.dataset_collection_dir

# Overwrite defaults, if destination path is given
if args.destination_dir:
    create_symlinks = True
    destination_dir = args.destination_dir
    os.makedirs(destination_dir, exist_ok=True)

for partition in ['train', 'test']:  # Handle both TORSO-21 partitions
    # Paths in dataset dir
    partition_dataset_dir = os.path.join(dataset_collection_dir, partition)
    partition_annotations_file = os.path.join(partition_dataset_dir, "annotations.yaml")
    segmentation_dir = os.path.join(partition_dataset_dir, "segmentations")

    # Assert paths exist
    assert os.path.exists(partition_dataset_dir), f"Is the given path correct? File or directory does not exist: '{partition_dataset_dir}'"
    assert os.path.exists(partition_annotations_file), f"Is the given path correct? File or directory does not exist: '{partition_annotations_file}'"
    assert os.path.exists(segmentation_dir), f"Is the given path correct? File or directory does not exist: '{segmentation_dir}'"

    # Paths in destination dir
    partition_destination_dir = os.path.join(destination_dir, partition)
    images_dir = os.path.join(partition_destination_dir, "images")
    labels_dir = os.path.join(partition_destination_dir, "labels")
    yoeo_segmentation_dir = os.path.join(partition_destination_dir, "yoeo_segmentations")

    # Create output directories if needed
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(yoeo_segmentation_dir, exist_ok=True)

    # Collect file names of images to dump them later to the train.txt or text.txt file
    image_names = []

    # Load annotation data from yaml file
    print(f"Loading annotation file: '{partition_annotations_file}'...")
    with open(partition_annotations_file, 'r') as f:
        data = yaml.safe_load(f)

    print("Writing outputs...")
    for img_name_with_extension, image_data in tqdm(data['images'].items()):
        image_names.append(img_name_with_extension)  # Collect image names
        img_name_without_extension = os.path.splitext(img_name_with_extension)[0]  # Remove file extension

        # SEGMENTATIONS
        ###############

        # Generate segmentations in correct format
        seg_in_path = os.path.join(segmentation_dir, img_name_without_extension + ".png")
        seg_in = cv2.imread(seg_in_path)
        if seg_in is not None:
            mask = np.zeros(seg_in.shape[:2], dtype=np.uint8)
            mask += ((seg_in == (127, 127, 127)).all(axis=2)).astype(np.uint8)  # Lines
            mask += (((seg_in >= (254, 254, 254)).all(axis=2)).astype(np.uint8) * 2)  # Field
            seg_out = np.zeros(seg_in.shape, dtype=np.uint8)
            seg_out[..., 0] = mask
            seg_out[..., 1] = mask
            seg_out[..., 2] = mask
            cv2.imwrite(os.path.join(yoeo_segmentation_dir, img_name_without_extension + ".png"), seg_out)
        else:
            print(f"No segmentation found: '{seg_in_path}'")
            continue

        # BOUNDING BOXES
        ################

        img_width = image_data['width']
        img_height = image_data['height']
        annotations = []

        for annotation in image_data['annotations']:
            # Skip annotations that are not in the image
            if not annotation['in_image']:
                continue

            # Derive the class name of the current annotation
            class_name = annotation['type']
            if args.robots_with_team_colors and class_name == 'robot':
                class_name += f"_{annotation['color']}"

            # Skip annotations, if is not a bounding box or should be skipped or is blurred or concealed and user chooses to skip them
            if (class_name in CLASSES['segmentation_classes'] or  # Handled by segmentations
                class_name in CLASSES['skip_classes'] or  # Skip this annotation class
                (args.skip_blurred and annotation.get('blurred', False)) or
                (args.skip_concealed and annotation.get('concealed', False))):
                continue
            elif class_name in CLASSES['bb_classes']:  # Handle bounding boxes
                min_x = min(map(lambda x: x[0], annotation['vector']))
                max_x = max(map(lambda x: x[0], annotation['vector']))
                min_y = min(map(lambda x: x[1], annotation['vector']))
                max_y = max(map(lambda x: x[1], annotation['vector']))

                annotation_width = max_x - min_x
                annotation_height = max_y - min_y
                relative_annotation_width = annotation_width / img_width
                relative_annotation_height = annotation_height / img_height

                center_x = min_x + (annotation_width / 2)
                center_y = min_y + (annotation_height / 2)
                relative_center_x = center_x / img_width
                relative_center_y = center_y / img_height

                # Derive classID from index in predefined classes
                classID = CLASSES['bb_classes'].index(class_name)                
                annotations.append(f"{classID} {relative_center_x} {relative_center_y} {relative_annotation_width} {relative_annotation_height}")
            else:
                print(f"The annotation type '{class_name}' is not supported. Image: '{img_name_with_extension}'")

        # Store bounding box annotations in .txt file
        with open(os.path.join(labels_dir, img_name_without_extension + ".txt"), "w") as output:
            output.writelines([annotation + "\n" for annotation in annotations])

    # Create symlinks for images to destination directory
    # This is helpful, if dataset directory is read-only
    if create_symlinks:
        for image_name in image_names:
            link_path = os.path.join(images_dir, image_name)
            target_path = os.path.join(partition_dataset_dir, "images", image_name)
            os.symlink(target_path, link_path)

    # Write train.txt or text.txt file containing full paths to each image
    partition_txt_path = os.path.join(destination_dir, partition, f"{partition}.txt")
    with open(partition_txt_path, "w") as partition_txt_file:
        partition_txt_file.writelines([str(os.path.join(images_dir, image_name)) + "\n" for image_name in image_names])

# The names file contains the class names of bb detections and segmentations
names_path = os.path.join(destination_dir, "yoeo_names.yaml")
names = {
    'detection': CLASSES['bb_classes'],
    'segmentation': CLASSES["segmentation_classes"],
}
with open(names_path, "w") as names_file:
    yaml.dump(names, names_file)

data_path = os.path.join(destination_dir, "yoeo.data")
with open(data_path, "w") as data_file:
    data_file.write(f"train={os.path.join(destination_dir, 'train', 'train.txt')}\n")
    data_file.write(f"valid={os.path.join(destination_dir, 'test', 'test.txt')}\n")
    data_file.write(f"names={names_path}\n")
