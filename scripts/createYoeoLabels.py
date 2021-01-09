#!/usr/bin/env python3

import os
import glob
import sys
import yaml
import cv2
import numpy as np

"""
Expected YAML format:
=====================

We use the ``Bit-Bots/YOEOyaml`` export format of the `Bit-Bots ImageTagger <https://imagetagger.bit-bots.de>`_.

set: <imageset_name>
images:
    <image_name>:
        width: <image_width>
        height: <image_height>
        annotations:
          - {type: <annotation_type>,
             blurred: <bool, is annotation blurred?>,
             concealed: <bool, is annotation concealed?>,
             vector:
                [
                    [x1,y1],
                    [x2,y2],
                ]}
          - {type: <annotation_type>,
             vector:
                [notinimage]}
    <next_image_name>:
        ...


Expects following directories:
==============================

Superset/
    - Dataset1/
        - images/
        - labels/
        - masks/
        - <annotation_file>.yaml
    - Dataset2/
        - images/
        - labels/
        - masks/
        - <annotation_file>.yaml
...
"""


if len(sys.argv[0]) == 0:
    superset = os.path.abspath(input("path to root of your datasets:"))
else:
    superset = os.path.abspath(sys.argv[1])

imagetagger_annotation_files = glob.glob(f"{superset}/*/*.yaml")
datasets = list(map(lambda x: os.path.basename(os.path.dirname(x)), imagetagger_annotation_files))

datasets_serialized = '\n'.join(datasets)
print(f"The following datasets will be considered: \n{datasets_serialized}")

trainImages = []  # this ensures only images with labels are used

# Iterate over all datasets
for yamlfile in imagetagger_annotation_files:
    d = os.path.dirname(yamlfile)

    print(f"Creating files for {os.path.basename(d)}\n")
    
    masks_dir = os.path.join(d, "masks")
    if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

    with open(yamlfile) as f:
        export = yaml.safe_load(f)

    for img_name, frame in export['images'].items():
        trainImages.append(os.path.join(d, "images", img_name))
        name = os.path.splitext(img_name)[0] # Remove extenion
        annolist = []
        for annotation in frame['annotations']:
            if not (annotation['vector'][0] == 'notinimage'):
                imgwidth = frame['width']
                imgheight = frame['height']
                if annotation['type'] in ["ball", "goalpost"]:
                    if not (annotation['vector'][0] == 'notinimage'):
                        min_x = min(map(lambda x: x[0], annotation['vector']))
                        max_x = max(map(lambda x: x[0], annotation['vector']))
                        min_y = min(map(lambda x: x[1], annotation['vector']))
                        max_y = max(map(lambda x: x[1], annotation['vector']))

                        annowidth = max_x - min_x
                        annoheight = max_y - min_y
                        relannowidth = annowidth / imgwidth
                        relannoheight = annoheight / imgheight

                        center_x = min_x + (annowidth / 2)
                        center_y = min_y + (annoheight / 2)
                        relcenter_x = center_x / imgwidth
                        relcenter_y = center_y / imgheight

                        if annotation['type'] == "ball":
                            classid = 0
                        if annotation['type'] == "goalpost":
                            classid = 1

                        annolist.append("{} {} {} {} {}".format(classid, relcenter_x, relcenter_y, relannowidth, relannoheight,))
                    else:
                        pass

                if annotation['type'] in ["field edge"]:

                    mask = np.zeros((imgheight, imgwidth, 3), dtype=np.uint8)

                    vector = [list(pts) for pts in list(annotation['vector'])]
                    vector.append([imgwidth - 1, imgheight - 0])
                    vector.append([0, imgheight - 1])  # extending the points to fill the space below the horizon
                    
                    points = np.array(vector, dtype=np.int32)
                    points = points.reshape((1, -1, 2))
                    cv2.fillPoly(mask, points, (1, 1, 1))

                    cv2.imwrite(os.path.join(masks_dir, name + ".png"), mask)

        label_dir = os.path.join(d, "labels")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        with open(os.path.join(label_dir, name + ".txt"), "w") as output:
            for line in annolist:
                output.write(line + "\n")

trainImages = set(trainImages) # prevent images from showing up twice
with open(os.path.join(superset, "train.txt"), "w") as traintxt:
    for e in trainImages:
        traintxt.write(e + "\n")
