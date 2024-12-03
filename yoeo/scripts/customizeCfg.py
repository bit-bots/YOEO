#!/usr/bin/env python3

"""
This file takes a given model architecture configuration file (cfg / toml)
and adapts it to the classes of a given dataset defined in a yaml file.
"""

import argparse
import yaml
from yoeo.utils.parse_config import (
    parse_model_config,
    write_model_config,
    parse_data_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customize a model architecture configuration file to a dataset"
    )
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        default="config/yoeo.cfg",
        help="Path to the model architecture configuration file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="config/custom.data",
        help="Path to the dataset configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="config/yoeo-custom.cfg",
        help="Path to the output model architecture configuration file",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    # Load the dataset configuration
    dataset_config = parse_data_config(args.dataset)

    # Load the class names from the dataset configuration
    with open(dataset_config["names"], "r") as f:
        class_names = yaml.safe_load(f)

    # Validate the dataset configuration
    assert (
        "detection" in class_names
    ), "Dataset configuration file must contain a 'detection' key listing all the object classes"
    assert isinstance(
        class_names["detection"], list
    ), "The 'detection' key in the dataset configuration file must be a list"
    assert (
        len(class_names["detection"]) > 0
    ), "The 'detection' key in the dataset configuration file must contain at least one class"
    assert all(
        isinstance(c, str) for c in class_names["detection"]
    ), "All classes in the 'detection' key must be strings"
    assert (
        "segmentation" in class_names
    ), "Dataset configuration file must contain a 'segmentation' key listing all the segmentation classes"
    assert isinstance(
        class_names["segmentation"], list
    ), "The 'segmentation' key in the dataset configuration file must be a list"
    assert (
        len(class_names["segmentation"]) > 0
    ), "The 'segmentation' key in the dataset configuration file must contain at least one class"
    assert all(
        isinstance(c, str) for c in class_names["segmentation"]
    ), "All classes in the 'segmentation' key must be strings"

    number_of_object_detection_classes = len(class_names["detection"])
    number_of_segmentation_classes = len(class_names["segmentation"])

    print(
        f"Found {number_of_object_detection_classes} object detection classes and {number_of_segmentation_classes} segmentation classes"
    )

    # Load the model configuration
    model_architecture = parse_model_config(args.cfg)

    # Search for all yolo layers in the model configuration and
    # adapt the number of classes as well as
    # the number of filters in the preceding convolutional layer
    for i, layer in enumerate(model_architecture):
        if layer["type"] == "yolo":
            # Adapt the number of classes
            layer["classes"] = number_of_object_detection_classes
            # Adapt the number of filters in the preceding convolutional layer
            assert (
                i > 0
            ), "Yolo layer can not be the first layer in the model architecture"
            prev_layer = model_architecture[i - 1]
            assert prev_layer.get("filters") is not None, (
                "Yolo layer must be preceded by a convolutional layer for this script to work, "
                "if you do more complex stuff, you have to adapt the configuration manually"
            )
            prev_layer["filters"] = (number_of_object_detection_classes + 5) * len(
                layer["mask"].split(",")
            )
        if layer["type"] == "seg":
            # Adapt the number of classes
            layer["classes"] = number_of_segmentation_classes
            # Adapt the number of filters in the preceding convolutional layer
            assert (
                i > 0
            ), "Seg layer can not be the first layer in the model architecture"
            prev_layer = model_architecture[i - 1]
            assert prev_layer.get("filters") is not None, (
                "Seg layer must be preceded by a convolutional layer for this script to work, "
                "if you do more complex stuff, you have to adapt the configuration manually"
            )
            prev_layer["filters"] = number_of_segmentation_classes

    # Write the adapted model configuration to the output file
    write_model_config(model_architecture, args.output)

    print(f"Model architecture adapted and saved to {args.output}")


if __name__ == "__main__":
    run()
