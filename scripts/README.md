# Scripts

## `createYOEOLabelsFromTORSO-21.py`

This script reads annotations in the expected yaml format below.
to generate the corresponding yolo .txt files and the segmentation masks.

### Expected YAML format (Example)

Please refer to the [TORSO-21 documentation](https://github.com/bit-bots/TORSO_21_dataset#structure) for this.

### Expects following file tree (Example)

We expect to be given a subdirectory of the structure documented [here](https://github.com/bit-bots/TORSO_21_dataset#structure):

```
<path-to-TORSO-21/reality OR path-to-TORSO-21/simulation>
├── train
│   ├── annotations.yaml
│   ├── images
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── segmentations
│       ├── image1.png
│       ├── image2.png
│       └── ...
└── test
    └── ... # Same as train
```

### Produces the following file tree (Example)

```
<destination-dir OR path-to-TORSO-21/reality OR path-to-TORSO-21/simulation>
├── train.txt
├── test.txt
├── yoeo.names
├── yoeo.data
├── train
│   ├── images  # Images already exist in dataset; symlinks are created in destination-dir case
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── labels
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── yoeo_segmentations
│       ├── image1.png
│       ├── image2.png
│       └── ...
└── test
    └── ... # Same as train
```

where:

- `train.txt` contains absolute image-paths for training
- `test.txt` contains absolute image-paths for evaluation
- `yoeo.names.yaml` contains names of bounding boxes and segmentation classes
- `yoeo.data` contains absolute paths to *train.txt*, *test.txt* and *yoeo.names*
