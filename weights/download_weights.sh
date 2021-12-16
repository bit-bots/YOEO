#!/bin/bash
# Download weights for yoeo-rev-7
wget "http://data.bit-bots.de/models/2021_12_06_flo_torso21_yoeo_7/yoeo.pth"
# Download weights for vanilla YOLOv3
wget -c "https://pjreddie.com/media/files/yolov3.weights" --header "Referer: pjreddie.com"
# # Download weights for tiny YOLOv3
wget -c "https://pjreddie.com/media/files/yolov3-tiny.weights" --header "Referer: pjreddie.com"
# Download weights for backbone network
wget -c "https://pjreddie.com/media/files/darknet53.conv.74" --header "Referer: pjreddie.com"
