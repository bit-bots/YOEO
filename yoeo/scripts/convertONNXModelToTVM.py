import argparse
import sys
from shutil import copyfile

import onnx

try:
    from tvm.driver import tvmc
    from tvm.driver.tvmc.model import TVMCModel
    from tvm.relay.transform import ToMixedPrecision
    from tvm import relay
except ModuleNotFoundError:
    print("Please install Apache TVM!")
    sys.exit(1)


def make_parser():
    parser = argparse.ArgumentParser("YOEO TVM Tuning")
    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the simplified onnx model")
    parser.add_argument(
        "--input_size",
        default=416,
        type=int,
        help="Input image size")
    parser.add_argument(
        "--trials",
        default=20000,
        type=int,
        help="Number of tuning trials")
    parser.add_argument(
        "--half",
        default=False,
        action="store_true",
        help="Tune with mixed precision instead of full precision (FP32)."
    )
    parser.add_argument(
        "--no_tuning",
        default=False,
        action="store_true",
        help="Disables the tuning of the model for testing purposes."
    )
    parser.add_argument(
        "-c",
        "--tuning_record_checkpoint",
        default=None,
        type=str,
        help="Prior tuning records")
    parser.add_argument(
        "-r",
        "--output_tuning_records",
        default="tuning_records.json",
        type=str,
        help="Path to the tuning records that are created for this optimization")
    parser.add_argument(
        "--target",
        default="vulkan -from_device=0",
        type=str,
        help="Target device (e.g. 'llvm' (CPU), 'vulkan' (GPU) or 'cuda' (GPU))")
    parser.add_argument(
        "-o",
        "--output_model",
        default="compiled_model.tar",
        type=str,
        help="File where the compiled model is saved after the tuning.")
    return parser


def run():
    # Make CLI
    args = make_parser().parse_args()

    # Define model input
    input_name = "InputLayer"
    shape_list = {input_name : (1, 3, args.input_size, args.input_size)}

    # Load onnx model into TVM
    onnx_model = onnx.load(args.onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_list)

    # Convert model to FP16 (half)
    if args.half:
        mod = ToMixedPrecision(mixed_precision_type='float16')(mod)

    # Build an TVM Compiler model
    tvmc_model = TVMCModel(mod, params)


    # Tune the model (depending on the hardware and parameters this takes days)
    if not args.no_tuning:
        tvmc.tune(
                tvmc_model,
                target=args.target,
                prior_records=args.tuning_record_checkpoint,
                tuning_records=args.output_tuning_records,
                min_repeat_ms=200,
                number=20,
                repeat=3,
                trials=args.trials,
                enable_autoscheduler=True
            )

    # Compile the model based on the optimizations discovered in the tuning
    package = tvmc.compile(
        tvmc_model,
        target=args.target,
        tuning_records=args.output_tuning_records)

    # Copy the temporary compiler result to the correct destination
    copyfile(package.package_path, args.output_model)


if __name__ == "__main__":
    run()
