import torch
from yoeo import models
import torchvision
from detectron2.modeling import build_model

def inference_func(model, image):
    inputs = [{"image": image}]
    return model.inference(inputs, do_postprocess=False)[0]

cfg = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/yoeo.cfg"
# weights = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/yoeo_mine_759.pth"
weights = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/yoeo.pth"
# print("cfg.MODEL.WEIGHTS: ",cfg.MODEL.WEIGHTS)   ## RETURNS : cfg.MODEL.WEIGHTS:  drive/Detectron2/model_final.pth

model = models.load_model(cfg, weights, "cpu")
print(model)

example = torch.rand(1, 3, 416, 416)
# wrapper = TracingAdapter(module, example, inference_func)
# model.eval()
traced_script_module = torch.jit.trace(model, example)
# traced_script_module = torch.jit.trace(wrapper, (example,))
traced_script_module.save("/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/model-final.pt")
