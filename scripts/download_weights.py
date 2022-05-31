#!/usr/bin/env python3
import os
import pathlib
import argparse
import urllib.request


def run():
    parser = argparse.ArgumentParser(description='Download YOEO pretrained weights')
    parser.add_argument(
        '--output',
        '-o',
        type=pathlib.Path,
        default='weights/yoeo.pth',
        help='The pretrained weights file (.pth) will be written to this path. Defaults to "weights/yoeo.pth"',
        )
    args = parser.parse_args()

    url = "https://data.bit-bots.de/models/2021_12_06_flo_torso21_yoeo_7/yoeo.pth"
    output_path = args.output

    with urllib.request.urlopen(url) as input_file:
        try:
            with open(output_path, 'xb') as output_file:
                print(f"Saving pretrained weights to: {output_path}")
                output_file.write(input_file.read())
        except FileExistsError as e:
            print(f"ERROR: The output file {output_path} does already exist. Will abort and not overwrite.")


if __name__ == '__main__':
    run()
