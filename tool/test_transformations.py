import os
import sys
import re
import argparse
import numpy as np
import torch
import copy
import PIL
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pytorch_lightning as pl

sys.path.append(".")
import datasets
from utils.box_ops import points_transformation, point_to_abs, point_to_rel
from utils.box_ops import (
    boxes_transformation,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    boxes_to_abs,
    boxes_to_rel,
    boxes_fit_size,
)


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def save_boxes_images(output_path, images, boxes=None, keypoints=None, mask=None, sizes=None, suffix=None):
    os.makedirs(output_path, exist_ok=True)

    assert len(images.shape) == 4, "Expects images with batch dimension"
    if mask is not None:
        assert len(mask.shape) == 3, "Expects images with batch dimension"

    for b in range(images.shape[0]):

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
        if boxes is not None:
            for i in range(boxes[b].shape[0]):
                box = boxes[b][i].cpu().numpy()
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
    pl.utilities.seed.seed_everything(42)

    strong_transforms = datasets.coco.make_coco_transforms("train", "strong")
    weak_transforms = datasets.coco.make_coco_transforms("train", "weak")
    # transforms2 = datasets.coco.make_coco_transforms("train", "weak")
    height = 400
    width = 600
    image = np.ones([400, 600, 3], dtype=np.uint8) * 255
    border = 150
    image[border, border, :] = [255, 0, 0]
    image[height - border, border, :] = [0, 255, 0]
    image[border, width - border, :] = [0, 0, 255]
    image[height - border, width - border, :] = [255, 0, 255]

    target = {
        "keypoints": torch.as_tensor(
            [
                [
                    [border, border, 2.0],
                    [border, height - border, 2.0],
                    [width - border, border, 2.0],
                    [width - border, height - border, 2.0],
                    [width / 2, height / 2, 2.0],
                ]
            ]
        ),
        "boxes": torch.as_tensor(
            [
                [border, border, width - border, height - border],
                # [border, border, width - border, height - 2 * border],
            ]
        ),
        "area": torch.as_tensor(
            [(width - 2 * border) * (height - 2 * border), (width - 2 * border) * (height - 3 * border)]
        ),
        "size": torch.as_tensor([height, width]),
    }
    for x in range(1):
        img_1, target_1 = weak_transforms(PIL.Image.fromarray(image), target)
        img_2, target_2 = strong_transforms(PIL.Image.fromarray(image), target)
        # print(target)
        # target_1["keypoints"][:, :, 0] /= width
        # target_1["keypoints"][:, :, 1] /= height
        # print(target_1)
        # print(image.shape)
        # print(img_1.shape)

        # img_1 = np.transpose(img_1.numpy(), (1, 2, 0))

        # norm_img_1 = 255 * (img_1 - np.amin(img_1)) / (np.amax(img_1) - np.amin(img_1))
        # fig, ax = plt.subplots()
        # ax.imshow((norm_img_1).astype(np.uint8))
        print(img_1)
        print([target_1["keypoints"]])
        save_boxes_images(
            args.output_path,
            torch.unsqueeze(img_1, dim=0),
            keypoints=[target_1["keypoints"]],
            suffix="_weak",
            sizes=[target_1["size"]],
            boxes=[target_1["boxes"]],
        )
        # exit()
        # img_2 = np.transpose(img_2.numpy(), (1, 2, 0))
        # print(img_2.shape)

        # norm_img_2 = 255 * (img_2 - np.amin(img_2)) / (np.amax(img_2) - np.amin(img_2))
        # fig, ax = plt.subplots()
        # ax.imshow((norm_img_2).astype(np.uint8))
        save_boxes_images(
            args.output_path,
            torch.unsqueeze(img_2, dim=0),
            keypoints=[target_2["keypoints"]],
            suffix="_strong",
            sizes=[target_2["size"]],
            boxes=[target_2["boxes"]],
        )
        # exit()

        # plt.savefig(os.path.join(args.output_path, f"00.png"), dpi=300)
        # plt.close(fig)squ

        inv_transformation = torch.linalg.inv(target_1["transformation"])
        print("######### POINTS #########")
        print(target["keypoints"])
        target_1_keypoints_abs = point_to_abs(target_1["keypoints"], target_1["size"])
        target_2_keypoints_abs = point_to_abs(target_2["keypoints"], target_2["size"])
        print(target_1_keypoints_abs)
        print(target_2_keypoints_abs)
        print("#####################")

        points_1_origin = points_transformation(target_1_keypoints_abs, inv_transformation)
        points_1_transformation_2_abs = points_transformation(points_1_origin, target_2["transformation"])
        print(points_1_origin)
        print(points_1_transformation_2_abs)
        print("++++++++++++++")
        points_1_transformation_2 = point_to_rel(points_1_transformation_2_abs, target_2["size"])
        print(point_to_rel(points_1_transformation_2_abs, target_2["size"]))
        print(target_2["keypoints"])

        print("######### BBOXES #########")
        print(target["boxes"])
        print(target_1["boxes"])
        boxes_1_abs = boxes_to_abs(box_cxcywh_to_xyxy(target_1["boxes"]), size=target_1["size"])
        print(boxes_1_abs)
        boxes_1_origin = boxes_transformation(boxes_1_abs, inv_transformation)
        print(boxes_1_origin)
        boxes_1_transformation_2_abs = boxes_transformation(boxes_1_origin, target_2["transformation"])
        print(boxes_1_transformation_2_abs)

        boxes_1_transformation_2_abs = boxes_fit_size(boxes_1_transformation_2_abs, size=target_2["size"])
        print(boxes_1_transformation_2_abs)
        print(target_2["size"])
        # exit()
        boxes_1_transformation_2 = box_xyxy_to_cxcywh(boxes_to_rel(boxes_1_transformation_2_abs, size=target_2["size"]))
        print(boxes_1_transformation_2_abs)
        print(boxes_to_abs(box_cxcywh_to_xyxy(target_2["boxes"]), size=target_2["size"]))

        save_boxes_images(
            args.output_path,
            torch.unsqueeze(img_2, dim=0),
            keypoints=[points_1_transformation_2],
            suffix="_transformed",
            sizes=[target_2["size"]],
            boxes=[boxes_1_transformation_2],
        )
        exit()

        for bbox_i in range(target_1["keypoints"].shape[0]):
            for p_i in range(target_1["keypoints"].shape[1]):
                point = target_1["keypoints"][bbox_i, p_i, :2]
                print(target_1["keypoints"][bbox_i, p_i, 2])
                if target_1["keypoints"][bbox_i, p_i, 2] < 2:
                    continue
                point = point * torch.as_tensor([target_1["size"][1], target_1["size"][0]])
                point = torch.cat([point, torch.as_tensor([1.0])])

                print("###############")
                print(target["keypoints"][0, p_i])
                print(point)
                weak_point = inv_transformation @ point
                print(weak_point)
                deltas = np.abs(weak_point[:2] - target["keypoints"][0, p_i, :2]).numpy()

                print(deltas)
                print(np.amax(deltas))

                if np.amax(deltas) > 0.001:
                    print("fail")
                    exit()
    exit()
    weak_image, weak_target, strong_image, strong_target = x

    save_boxes_images(
        args.output_path,
        weak_image.tensors,
        bbox=[y["boxes"] for y in weak_target],
        keypoints=[y["keypoints"] for y in weak_target],
        mask=weak_image.mask,
        suffix="_weak",
        sizes=[y["size"] for y in weak_target],
    )

    new_keypoints = []
    for strong, weak in zip(strong_target, weak_target):

        inv_transformation = torch.linalg.inv(strong["transformation"])
        transformation = weak["transformation"] @ inv_transformation
        keypoints_copy = copy.deepcopy(strong["keypoints"])
        for bbox_i in range(strong["keypoints"].shape[0]):
            for p_i in range(strong["keypoints"].shape[1]):
                point = strong["keypoints"][bbox_i, p_i, :2]
                point = point * torch.tensor(strong["size"], dtype=torch.float32)
                point = torch.cat([point, torch.as_tensor([1.0])])

                weak_point = transformation @ point
                keypoints_copy[bbox_i, p_i, :2] = weak_point[:2] / torch.tensor(weak["size"], dtype=torch.float32)

        new_keypoints.append(keypoints_copy)

    save_boxes_images(
        args.output_path,
        weak_image.tensors,
        bbox=[y["boxes"] for y in weak_target],
        keypoints=new_keypoints,
        mask=weak_image.mask,
        suffix="_transformed",
        sizes=[y["size"] for y in weak_target],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
