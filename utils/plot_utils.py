"""
Plotting utilities to visualize training logs.
"""
from pickletools import uint8
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import imageio
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cv2


def plot_logs(logs, fields=("class_error", "loss_bbox_unscaled", "mAP"), ewm_col=0, log_name="log.txt"):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}"
            )

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == "mAP":
                coco_eval = pd.DataFrame(np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f"train_{field}", f"test_{field}"], ax=axs[j], color=[color] * 2, style=["-", "--"]
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_prediction_images_matplotlib(
    image, output_path=None, boxes=None, keypoints=None, keypoints_labels=None, keypoints_scores=None, min_scores=0.9
):

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


def plot_prediction_images_cv2(
    image, output_path=None, boxes=None, keypoints=None, keypoints_labels=None, keypoints_scores=None, min_scores=0.9
):
    plot = image.copy()
    # fig, ax = plt.subplots(frameon=False)
    # ax.set_axis_off()
    # tab10
    color_lut = [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [152, 223, 138],
        [255, 152, 150],
        [196, 156, 148],
        [197, 176, 213],
        [199, 199, 199],
        [219, 219, 141],
        [158, 218, 229],
    ]

    bones_lut = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]
    # ax.imshow(image)
    if boxes is not None:
        for i, box in enumerate(boxes):
            box = np.asarray(box)
            box[box < 0] = 0
            box = np.rint(box).astype(np.uint)
            cv2.rectangle(plot, box[0:2], box[2:4], color=color_lut[i % len(color_lut)], thickness=2)
            # rect = Rectangle(
            #     [box[0], box[1]],
            #     box[2] - box[0],
            #     box[3] - box[1],
            #     linewidth=1,
            #     edgecolor="r",
            #     facecolor="none",
            # )
            # ax.add_patch(rect)

    if keypoints is not None:
        # print(keypoints)
        # print(keypoints_labels)
        # print(keypoints_scores)
        # exit()
        for j, (pose, labels) in enumerate(zip(keypoints, keypoints_labels)):
            point_lut = {}
            pose = np.rint(np.asarray(pose)).astype(np.uint)
            for i, (keypoint, label) in enumerate(zip(pose, labels)):
                if keypoints_scores is not None:
                    if keypoints_scores[j][i] < min_scores:
                        continue

                point_lut[label.numpy().item()] = keypoint
                # print(f"{keypoint} {label}")
                cv2.circle(plot, keypoint, radius=2, color=color_lut[label], thickness=2)

            for i, bone in enumerate(bones_lut):
                if bone[0] - 1 not in point_lut or bone[1] - 1 not in point_lut:
                    continue
                cv2.line(plot, point_lut[bone[0] - 1], point_lut[bone[1] - 1], color=color_lut[i], thickness=2)

    imageio.imwrite(output_path, plot)
    # plt.savefig(output_path, dpi=300)
    # plt.close(fig)


def plot_prediction_images(
    image, output_path=None, boxes=None, keypoints=None, keypoints_labels=None, keypoints_scores=None, backend="cv2"
):
    if backend == "matplotlib":
        plot_prediction_images_matplotlib(
            image, output_path, boxes, keypoints, keypoints_labels=keypoints_labels, keypoints_scores=keypoints_scores
        )
    if backend == "cv2":
        plot_prediction_images_cv2(
            image, output_path, boxes, keypoints, keypoints_labels=keypoints_labels, keypoints_scores=keypoints_scores
        )


def plot_precision_recall(files, naming_scheme="iter"):
    if naming_scheme == "exp_id":
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == "iter":
        names = [f.stem for f in files]
    else:
        raise ValueError(f"not supported {naming_scheme}")
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data["precision"]
        recall = data["params"].recThrs
        scores = data["scores"]
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data["recall"][0, :, 0, -1].mean()
        print(
            f"{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, "
            + f"score={scores.mean():0.3f}, "
            + f"f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}"
        )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title("Precision / Recall")
    axs[0].legend(names)
    axs[1].set_title("Scores / Recall")
    axs[1].legend(names)
    return fig, axs