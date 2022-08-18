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


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-p", "--prediction_path", help="verbose output")
    parser.add_argument("-a", "--annotation_path", help="verbose output")
    parser.add_argument("-s", "--skip_missing", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    prediction_dict = {}
    with open(args.prediction_path, "r") as f:
        for line in f:
            # print(line)
            d = json.loads(line)
            prediction_dict[d["id"]] = d

    with open(args.annotation_path, "r") as f:
        with open(args.output_path, "w") as f_out:
            for line in f:
                d = json.loads(line)
                if d["id"] not in prediction_dict:
                    if args.skip_missing:
                        continue
                    else:
                        f_out.write(json.dumps(d) + "\n")
                else:

                    results = {**prediction_dict[d["id"]], **d}
                    f_out.write(json.dumps(results) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())