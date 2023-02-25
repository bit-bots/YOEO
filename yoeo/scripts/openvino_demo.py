######
# Openvino stuff
######

# Commands to convert the onnx model to a OpenVino one
# source /opt/intel/openvino_2021/bin/setupvars.sh
# python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py --input_model ~/YOEO/onnxseg.onnx --framework onnx

# Demo Segmentation OpenVino Code

import time
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

# Load Network
ie = IECore()
net = IENetwork(model="~/onnxseg.xml", weights="~/onnxseg.bin")

# Push it to the device
# Device type
device = "MYRIAD"
exec_net = ie.load_network(network=net, num_requests=2, device_name=device)

# Read image file
img = cv2.imread("~/demo_img.png")

# Prepare image
img_tensor = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (416,416)).astype(np.float) / 255

# Run and profile network on the device
t1 = time.time()
exec_net.start_async(request_id=1, inputs={"InputLayer": img_tensor.transpose(2, 0, 1)})
exec_net.requests[1].wait(-1)
print(time.time() - t1)
# E.g. 0.15500497817993164

# Extract the mask and scale it so it can the saved as a image
mask = cv2.cvtColor((exec_net.requests[1].output_blobs['Segmentations'].buffer * 127).astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_GRAY2RGB)

# Show and save segmentation mask
cv2.imshow("test", mask); cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("/tmp/ncs_pred1.png", mask)

