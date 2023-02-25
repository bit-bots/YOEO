#! /usr/bin/env python3
import argparse
import os


def convert_model(onnx_path: str, output_path: str) -> None:
    command = assemble_command(onnx_path, output_path)
    run_command(command)


def assemble_command(onnx_path: str, output_path: str) -> str:
    # https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html (April 7, 2022)
    mo_command = f"""mo
		         --input_model "{onnx_path}"
		         --output_dir "{output_path}"
		         --input InputLayer
		         --output Detections,Segmentations
		         --framework onnx
		         --static_shape
		         --batch 1
		         """
    mo_command = " ".join(mo_command.split())

    return mo_command


def run_command(command: str) -> None:
    # https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html (April 7, 2022)
    print("Exporting ONNX model to IR...")

    mo_result = os.system(command)

    print("=" * 30)
    if mo_result == 0:
        print("Model conversion was successful")
    else:
        print("Model conversion failed")


def get_output_path(model_onnx: str) -> str:
    return get_parent_dir(model_onnx)


def get_parent_dir(path: str) -> str:
    absolute_path = os.path.abspath(path)
    parent_dir, filename = os.path.split(absolute_path)

    return parent_dir


def run():
    parser = argparse.ArgumentParser(description='Convert ONNX Model to OpenVino IR')
    parser.add_argument(
        "model_onnx",
        type=str,
        help="full path to model file (.onnx)"
    )

    args = parser.parse_args()

    output_path = get_output_path(args.model_onnx)
    convert_model(args.model_onnx, output_path)


if __name__ == "__main__":
    run()
