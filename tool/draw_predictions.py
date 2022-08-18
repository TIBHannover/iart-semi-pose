import os
import sys
import re
import argparse
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio
import json

import pytorch_lightning as pl

sys.path.append(".")

from utils.box_ops import points_transformation, point_to_abs, point_to_rel

from datasets import DatasetsManager


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    parser.add_argument("-i", "--image_path", help="verbose output")
    parser.add_argument("-b", "--boxes")
    parser.add_argument("-k", "--keypoints")
    parser.add_argument("-t", "--threshold", default=0.0)
    parser.add_argument("-s", "--scores")
    args = parser.parse_args()
    return args


def save_prediction_images(output_path, image, boxes=None, keypoints=None, scores=None, threshold=0.0):

    fig, ax = plt.subplots(frameon=False)
    ax.set_axis_off()

    ax.imshow(image)
    if boxes is not None:
        for box in boxes:

            rect = Rectangle(
                [box[0], box[1]],
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

    if keypoints is not None:
        for pose in keypoints:
            pose = np.asarray(pose)
            plt.scatter(pose[:, 0], pose[:, 1], marker=".")

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    image = imageio.imread(args.image_path)

    boxes = None
    if args.boxes is not None:
        boxes = json.loads(args.boxes)

    keypoints = None
    if args.keypoints is not None:
        keypoints = json.loads(args.keypoints)

    scores = None
    if args.scores is not None:
        scores = json.loads(args.scores)

    if args.scores is not None:
        scores = json.loads(args.scores)

    save_prediction_images(
        args.output_path, image, boxes=boxes, keypoints=keypoints, scores=scores, threshold=args.threshold
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
