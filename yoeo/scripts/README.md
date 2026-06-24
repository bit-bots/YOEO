# Scripts

## `createYOEOLabelsFromCOCO.py`

This script reads a COCO 1.0 style annotation file (e.g. exported from CVAT) and generates the corresponding YOEO `.txt` label files and segmentation masks, including a seeded train/test split. The source dataset is only ever read from, never modified.

It currently only extracts:
- Detection boxes for the `ball` and `robot` categories.
- Segmentation masks for `field lines` and `football field`, painted on top of an implicit `background`. Lines are painted on top of the field, so they remain visible even where the two overlap.

Any other COCO categories (e.g. `human`, `goalpost`) are ignored.

### Example usage

```bash
./createYOEOLabelsFromCOCO.py /path/to/dataset /path/to/output-dir
```

- Use a different train/test split ratio or seed:

```bash
./createYOEOLabelsFromCOCO.py /path/to/dataset /path/to/output-dir --test-split 0.1 --seed 1337
```

- Get help and information about arguments:

```bash
./createYOEOLabelsFromCOCO.py --help
```

### Expects following file tree (Example)

```
<dataset-dir>
├── annotations
│   └── instances_default.json
└── images
    ├── image1.png
    ├── image2.png
    └── ...
```

### Produces the same output file tree as `createYOEOLabelsFromTORSO-21.py` (see below), with images symlinked into the destination directory.

## `viewYOEODataset.py`

A small viewer to sanity-check a YOEO formatted dataset (bounding boxes + segmentation masks) without running any training or inference.

### Example usage

- Save annotated samples to disk (works headless):

```bash
./viewYOEODataset.py /path/to/yoeo.data --partition train --num-samples 20 --output-dir /tmp/yoeo-preview
```

- Interactively step through samples in a window (`n`/space: next, `p`: previous, `q`/Esc: quit):

```bash
./viewYOEODataset.py /path/to/yoeo.data --show
```

## `createYOEOLabelsFromTORSO-21.py`

This script reads annotations in the expected yaml format (see down below) to generate the corresponding yolo `.txt` files and the segmentation masks.

## Example usage

*NOTE: Replace paths in the following examples for your needs!*

- Create labels for TORSO-21 reality dataset:

```bash
./createYOEOLabelsFromTORSO-21.py <path-to-TORSO-21/reality>
```

- Create labels for TORSO-21 simulation dataset and write to custom output-path:

```bash
./createYOEOLabelsFromTORSO-21.py /path/to/TORSO-21/simulation --destination-dir /path/to/output-dir
```

- Get help and information about arguments:

```bash
./createYOEOLabelsFromTORSO-21.py --help
```

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
