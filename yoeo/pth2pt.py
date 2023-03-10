import torch
import models
import torchvision
from detectron2.export.flatten import TracingAdapter

cfg = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/yoeo.cfg"
weights = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/yoeo_mine_759.pth"
# print("cfg.MODEL.WEIGHTS: ",cfg.MODEL.WEIGHTS)   ## RETURNS : cfg.MODEL.WEIGHTS:  drive/Detectron2/model_final.pth

model = models.load_model(cfg, weights)
# print(model)
# model.eval()
print(torch.__version__)
example = torch.rand(1, 3, 416, 416)
print(f"type(example): {type(example)}")
# wrapper = TracingAdapter(module, example, inference_func)
# wrapper.eval()
# traced_script_module = torch.jit.script(model)
print(f"TYT KOSYAK")
traced_script_module = torch.jit.trace(model, example)#, check_trace=False)
# traced_script_module = torch.jit.trace(wrapper, (example,))
traced_script_module.save("/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/ochko.pt")
# model.save_darknet_weights("/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/save_dark_w.pt")
print("YEST")
