#!/usr/bin/env python3

"""Step through a YOEO formatted dataset (bounding boxes + segmentation masks) for a sanity check, without running any training or inference."""

import argparse
import os

import cv2
import numpy as np
import yaml

from yoeo.utils.parse_config import parse_data_config

BOX_COLOR = (0, 0, 255)
SEGMENTATION_COLORS = [
    (0, 0, 0),      # background
    (255, 255, 255),  # lines
    (0, 200, 0),    # field
]
DEFAULT_COLOR = (255, 0, 255)


def load_class_names(names_path):
    with open(names_path, 'r') as f:
        names = yaml.safe_load(f)
    return names['detection'], names['segmentation']


def load_image_list(list_path):
    with open(list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def resolve_label_path(image_path):
    image_dir = os.path.dirname(image_path)
    label_dir = "labels".join(image_dir.rsplit("images", 1))
    return os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")


def resolve_mask_path(image_path):
    image_dir = os.path.dirname(image_path)
    mask_dir = "yoeo_segmentations".join(image_dir.rsplit("images", 1))
    return os.path.join(mask_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")


def render_sample(image_path, detection_classes, alpha=0.4):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'")
    height, width = img.shape[:2]

    overlay = img.copy()
    mask_path = resolve_mask_path(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        color_mask = np.zeros_like(img)
        for class_id in np.unique(mask):
            color = SEGMENTATION_COLORS[class_id] if class_id < len(SEGMENTATION_COLORS) else DEFAULT_COLOR
            color_mask[mask == class_id] = color
        overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
    else:
        print(f"No segmentation mask found: '{mask_path}'")

    label_path = resolve_label_path(image_path)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id, cx, cy, w, h = line.split()
                class_id = int(class_id)
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)

                box_width, box_height = w * width, h * height
                x1 = int(cx * width - box_width / 2)
                y1 = int(cy * height - box_height / 2)
                x2 = int(cx * width + box_width / 2)
                y2 = int(cy * height + box_height / 2)

                cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOR, 2)
                label = detection_classes[class_id] if class_id < len(detection_classes) else str(class_id)
                cv2.putText(overlay, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
    else:
        print(f"No label file found: '{label_path}'")

    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=str, help="Path to a yoeo '.data' file (as produced by the createYOEOLabelsFrom* scripts).")
    parser.add_argument("--partition", choices=["train", "valid"], default="train", help="Which partition to look at.")
    parser.add_argument("--num-samples", type=int, default=None, help="Only look at the first N samples instead of the whole partition.")
    parser.add_argument("--output-dir", type=str, default=None, help="If given, annotated images are written here.")
    parser.add_argument("--show", action="store_true",
                         help="Open an interactive window to step through samples ('n'/space next, 'p' previous, 'q'/Esc quit).")
    args = parser.parse_args()

    if not args.output_dir and not args.show:
        parser.error("Nothing to do: pass --output-dir and/or --show.")

    options = parse_data_config(args.data_file)
    image_paths = load_image_list(options[args.partition])
    if args.num_samples:
        image_paths = image_paths[:args.num_samples]

    detection_classes, _ = load_class_names(options['names'])

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    index = 0
    while 0 <= index < len(image_paths):
        image_path = image_paths[index]
        overlay = render_sample(image_path, detection_classes)

        if args.output_dir:
            out_path = os.path.join(args.output_dir, os.path.basename(image_path))
            cv2.imwrite(out_path, overlay)
            print(f"[{index + 1}/{len(image_paths)}] Wrote '{out_path}'")

        if args.show:
            cv2.imshow("YOEO dataset viewer", overlay)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('p'):
                index = max(0, index - 1)
                continue
            else:
                index += 1
        else:
            index += 1

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
