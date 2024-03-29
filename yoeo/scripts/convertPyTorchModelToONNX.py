#! /usr/bin/env python3
import argparse
import os.path
from typing import Tuple

import onnx
import torch

import yoeo.models


def convert_model(model_cfg: str, weights_pth: str, output_path: str) -> None:
    pytorch_model = yoeo.models.load_model(model_cfg, weights_pth)
    convert_to_onnx(model=pytorch_model, output_path=output_path)


def convert_to_onnx(model: yoeo.models.Darknet, output_path: str, image_size: int = 416, batch_size: int = 1) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        export_params=True,
        input_names=["InputLayer"],
        output_names=["Detections", "Segmentations"],
        opset_version=11
    )


def check_model(model_path: str) -> None:
    onnx_model = load_onnx(model_path)
    check_onnx(onnx_model)


def load_onnx(path: str) -> onnx.onnx_ml_pb2.ModelProto:
    return onnx.load(path)


def check_onnx(model: onnx.onnx_ml_pb2.ModelProto) -> None:
    # https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md (April 7, 2022)
    print("="*30)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')


def construct_path(model_cfg: str) -> str:
    parent_dir = get_parent_dir(model_cfg)
    filename = get_filename_wout_extension(model_cfg)

    onnx_path = os.path.join(parent_dir, f"{filename}.onnx")

    return onnx_path


def get_parent_dir(path: str) -> str:
    absolute_path = os.path.abspath(path)
    parent_dir, filename = os.path.split(absolute_path)

    return parent_dir


def get_filename_wout_extension(path: str) -> str:
    absolute_path = os.path.abspath(path)
    parent_dir, filename = os.path.split(absolute_path)
    filename, ext = os.path.splitext(filename)

    return filename


def run():
    parser = argparse.ArgumentParser(description='Convert PyTorch Model to ONNX')
    parser.add_argument(
        "model_cfg",
        type=str,
        help="full path to model file (.cfg). ONNX model will be output with the same filename as well."
    )
    parser.add_argument(
        "model_weights",
        type=str,
        help="full path to model weights file (.pth or .weights)."
    )

    args = parser.parse_args()

    onnx_path = construct_path(args.model_cfg)
    convert_model(args.model_cfg, args.model_weights, onnx_path)
    check_model(onnx_path)


if __name__ == "__main__":
    run()
