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

from datasets import DatasetsManager


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    parser = DatasetsManager.add_args(parser)
    args = parser.parse_args()
    return args


def save_bbox_images(output_path, images, bbox=None, keypoints=None, mask=None, sizes=None, suffix=None, names=None):
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

        if names is None:
            plt.savefig(os.path.join(output_path, f"{b}{suffix}.png"), dpi=300)
        else:
            plt.savefig(os.path.join(output_path, f"{names[b]}_{b}{suffix}.png"), dpi=300)

        plt.close(fig)


def transform_prediction(weak_outputs, weak_targets, strong_targets):
    pass


def _build_unsupervised_target(weak_outputs, weak_targets, strong_targets):

    unsupervised_target = {"boxes": [], "labels": [], "neg_boxes": [], "size": strong_targets["size"]}
    # print(supervised["target"])
    # exit()

    batch_size = weak_outputs["pred_logits"].shape[0]
    # background_label = weak_outputs["pred_logits"].shape[-1] - 1
    # print(f"Background {background_label}")

    # print(weak_outputs)
    # print(weak_outputs['pred_logits'].shape)
    label_softmax = torch.softmax(weak_outputs["pred_logits"], dim=-1)
    top_prediction = label_softmax > 0.9
    boxes_pos = top_prediction[..., :-1].nonzero()
    boxes_neg = top_prediction[..., -1].nonzero()

    # exit()
    # transform_cordinates
    for b in range(batch_size):
        boxes = []
        neg_boxes = []
        labels = []
        inv_transformation = torch.linalg.inv(weak_targets["transformation"][b])
        weak_boxes_abs = boxes_to_abs(box_cxcywh_to_xyxy(weak_outputs["pred_boxes"][b]), size=weak_targets["size"][b])
        boxes_origins_abs = boxes_transformation(weak_boxes_abs, inv_transformation)

        strong_boxes_transformation_abs = boxes_transformation(boxes_origins_abs, strong_targets["transformation"][b])

        strong_boxes_transformation_abs = boxes_fit_size(
            strong_boxes_transformation_abs, size=strong_targets["size"][b]
        )

        strong_boxes_transformation = box_xyxy_to_cxcywh(
            boxes_to_rel(strong_boxes_transformation_abs, size=strong_targets["size"][b])
        )

        strong_boxes_transformation[b]
        boxes_sample = boxes_pos[boxes_pos[:, 0] == b]
        neg_boxes_sample = boxes_neg[boxes_neg[:, 0] == b]

        # we sample some hard negativ prediction
        for box in neg_boxes_sample.unbind(0):
            box_index = box[1]
            box_cxcywh = strong_boxes_transformation[box_index]
            neg_boxes.append(box_cxcywh)

        for box in boxes_sample.unbind(0):
            box_index = box[1]
            box_cls = box[2]
            box_cxcywh = strong_boxes_transformation[box_index]
            if [1] is None:
                labels.append(box_cls)
                boxes.append(box_cxcywh)
            else:
                box_label = box_cls.detach().cpu().numpy().item()
                if box_label in [1]:

                    # print(box_label)
                    labels.append(box_cls)
                    boxes.append(box_cxcywh)

                # exit()
        #     print(f"{box} {box_index} {box_cls} {box_cxcywh}")
        # print(boxes)
        # print(labels)
        if len(boxes) > 0:
            unsupervised_target["boxes"].append(torch.stack(boxes, dim=0))
            unsupervised_target["labels"].append(torch.stack(labels, dim=0))
        else:
            unsupervised_target["boxes"].append(torch.zeros([0, 4], device=label_softmax.device))
            unsupervised_target["labels"].append(torch.zeros([0], dtype=torch.int64, device=label_softmax.device))

        if len(neg_boxes) > 0:
            unsupervised_target["neg_boxes"].append(torch.stack(neg_boxes, dim=0))
        else:
            unsupervised_target["neg_boxes"].append(torch.zeros([0, 4], device=label_softmax.device))

    return unsupervised_target


def main():
    args = parse_args()

    pl.seed_everything(1337)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)
    print("start")

    for i, batch in enumerate(dataset.train()):
        # print(x.keys())
        supervised = batch["supervised"]
        samples, targets = supervised["image"], supervised["target"]
        unsupervised = batch["unsupervised"]
        weak_samples, weak_targets, strong_samples, strong_targets = (
            unsupervised["weak_image"],
            unsupervised["weak_target"],
            unsupervised["strong_image"],
            unsupervised["strong_target"],
        )
        # print(type(image))

        save_bbox_images(
            args.output_path,
            samples.tensors,
            bbox=[y for y in targets["boxes"]],
            # keypoints=[y for y in targets["keypoints"]],
            mask=samples.mask,
            suffix="",
            sizes=[y for y in targets["size"]],
            names=[x.numpy().item() for x in targets["image_id"]],
        )
        exit()
        # save_bbox_images(
        #     args.output_path,
        #     weak_image.tensors,
        #     bbox=[y["boxes"] for y in weak_target],
        #     keypoints=[y["keypoints"] for y in weak_target],
        #     mask=weak_image.mask,
        #     suffix="_weak",
        #     sizes=[y["size"] for y in weak_target],
        # )

        # new_keypoints = []
        # for strong, weak in zip(strong_target, weak_target):

        #     inv_transformation = torch.linalg.inv(strong["transformation"])
        #     transformation = weak["transformation"] @ inv_transformation
        #     keypoints_copy = copy.deepcopy(strong["keypoints"])
        #     for bbox_i in range(strong["keypoints"].shape[0]):
        #         for p_i in range(strong["keypoints"].shape[1]):
        #             point = strong["keypoints"][bbox_i, p_i, :2]
        #             point = point * torch.tensor(strong["size"], dtype=torch.float32)
        #             point = torch.cat([point, torch.as_tensor([1.0])])

        #             weak_point = transformation @ point
        #             keypoints_copy[bbox_i, p_i, :2] = weak_point[:2] / torch.tensor(weak["size"], dtype=torch.float32)

        #     new_keypoints.append(keypoints_copy)

        # save_bbox_images(
        #     args.output_path,
        #     weak_image.tensors,
        #     bbox=[y["boxes"] for y in weak_target],
        #     keypoints=new_keypoints,
        #     mask=weak_image.mask,
        #     suffix="_transformed",
        #     sizes=[y["size"] for y in weak_target],
        # )

    return 0


if __name__ == "__main__":
    sys.exit(main())
