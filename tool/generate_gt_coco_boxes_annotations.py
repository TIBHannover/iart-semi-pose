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
    parser.add_argument("-a", "--annotation_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coco = COCO(args.annotation_path)

    ids = list(sorted(coco.imgs.keys()))

    data_dict = {}
    for id in ids:
        path = coco.loadImgs(id)[0]["file_name"]
        annotations = coco.loadAnns(coco.getAnnIds(id))
        for a in annotations:
            if a["image_id"] not in data_dict:
                data_dict[a["image_id"]] = {
                    "id": a["image_id"],
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                    "path": path,
                }

            data_dict[a["image_id"]]["labels"].append(1)
            data_dict[a["image_id"]]["scores"].append(1.0)
            x, y, w, h = a["bbox"]
            data_dict[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        # f_out.write(json.dumps({**d, "path": path, "labels": [1],'scores':[1.0]}) + "\n")

    with open(args.output_path, "w") as f_out:
        for k, v in data_dict.items():
            f_out.write(json.dumps(v) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())