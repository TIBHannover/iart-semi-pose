import os
import sys
import re
import argparse
import json
import copy

import torch
import numpy as np

sys.path.append(".")
from utils.box_ops import box_iou, points_in_box

from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-p", "--prediction_path", help="verbose output")
    parser.add_argument("-a", "--annotation_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coco = COCO(args.annotation_path)

    with open(args.prediction_path, "r") as f:
        with open(args.output_path, "w") as f_out:
            for line in f:
                d = json.loads(line)
                path = coco.loadImgs(d["id"])[0]["file_name"]
                f_out.write(json.dumps({**d, "path": path}) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())