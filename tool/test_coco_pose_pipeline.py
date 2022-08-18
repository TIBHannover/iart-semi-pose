import os
import sys
import re
import argparse
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pytorch_lightning as pl

sys.path.append(".")

from utils.box_ops import points_transformation, point_to_abs, point_to_rel

from datasets import DatasetsManager


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    parser = DatasetsManager.add_args(parser)
    args = parser.parse_args()
    return args


def save_bbox_images(output_path, images, bbox=None, keypoints=None, mask=None, sizes=None, suffix=None):
    os.makedirs(output_path, exist_ok=True)

    assert len(images.shape) == 4, "Expects images with batch dimension"
    if mask is not None:
        assert len(mask.shape) == 3, "Expects images with batch dimension"

    for b in range(images.tensors.shape[0]):

        image = np.transpose(images[b].cpu().numpy(), (1, 2, 0))

        norm_image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
        if sizes is None:
            height = norm_image.shape[0].cpu().numpy()
            width = norm_image.shape[1].cpu().numpy()
        else:
            height = sizes[b][0].cpu().numpy()
            width = sizes[b][1].cpu().numpy()
        fig, ax = plt.subplots()
        ax.imshow((norm_image * 255).astype(np.uint8))
        if bbox is not None:
            for i in range(bbox[b].shape[0]):
                box = bbox[b][i].cpu().numpy()
                rect = Rectangle(
                    ((box[0] - box[2] / 2) * width, (box[1] - box[3] / 2) * height),
                    box[2] * width,
                    box[3] * height,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
        if keypoints is not None:
            for i in range(keypoints[b].shape[0]):
                keypoint = keypoints[b][i].cpu().numpy()
                for k in range(keypoint.shape[0]):
                    if keypoint[k, 2] > 1:
                        plt.scatter(keypoint[k, 0] * width, keypoint[k, 1] * height)

        if suffix is None:
            suffix = ""

        plt.savefig(os.path.join(output_path, f"{b}{suffix}.png"), dpi=300)
        plt.close(fig)


def main():
    args = parse_args()

    pl.seed_everything(42)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)
    print("start")
    for i, x in enumerate(dataset.train()):
        for b in range(x["image"].tensors.shape[0]):
            print(x["box_target"]["image_id"][b])
            targets = x["box_target"]
            image = np.transpose(x["image"].tensors[b].cpu().numpy(), (1, 2, 0))

            norm_image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))

            fig, ax = plt.subplots()
            ax.imshow((norm_image * 255).astype(np.uint8))
            plt.savefig(os.path.join(args.output_path, f"{x['box_target']['image_id'][b]}_{b}_pose.png"), dpi=300)
            plt.close(fig)

            inv_transformation = torch.linalg.inv(targets["transformation"][b])
            coords_abs = point_to_abs(targets["joints"][b], targets["size"][b])
            coords_origin_abs = points_transformation(coords_abs, inv_transformation)
            print(coords_origin_abs)

        exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
