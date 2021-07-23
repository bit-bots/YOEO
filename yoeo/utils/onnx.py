import onnx
#import onnxruntime
import torch

from yoeo import models

def convert_to_onnx(model, image_size, batch_size, output_path="yoeo.onnx"):
    dummy_input = torch.randn(batch_size,3,image_size,image_size, device="cuda")
    torch.onnx.export(
        model.cuda(),
        dummy_input,
        output_path,
        verbose=True,
        export_params=True,
        input_names=["InputLayer"],
        output_names=["YOLODetections", "Segmentations"],
        opset_version=11)

def check_onnx(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)

def _to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def run_onnx(img, onnx_model_path="yoeo.onnx"):
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: _to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0], ort_outs[1]
