import argparse
import json
import os
import random
import logging
import copy
from typing import Any, Callable, Optional, Tuple, List, Dict

import torch
from PIL import Image

import utils.misc as utils
from datasets import DatasetsManager
from datasets.coco_iterator import CocoIteratorDataloader, CocoIteratorDataset
from datasets.coco import make_coco_transforms, get_node_rank
from datasets.pipeline import (
    MergePipeline,
    Pipeline,
    SequencePipeline,
    RepeatPipeline,
    ConcatShufflePipeline,
    split_chunk_by_workers,
)

from datasets.transforms import crop
from utils.box_ops import box_xyxy_to_cxcywh, boxes_scale, boxes_aspect_ratio

import datasets.transforms as T


class CocoPoseIteratorDataset(CocoIteratorDataset):
    def __init__(self, *args, scale_boxes=None, min_keypoint_visible=1.0, **kwargs):
        super(CocoPoseIteratorDataset, self).__init__(*args, **kwargs)
        self.scale_boxes = scale_boxes
        if scale_boxes is None:
            self.scale_boxes = 1.0

        self.target_height = 384
        self.target_width = 288
        self.ar = self.target_width / self.target_height

        self.min_keypoint_visible = min_keypoint_visible

        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.keypoints = {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle",
        }

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def hflip_annotations(self, keypoints):

        flip_lut = {}
        for x in range(len(self.keypoints)):
            flip_lut[x] = x

        for pair in self.flip_pairs:
            flip_lut[pair[0]] = pair[1]
            flip_lut[pair[1]] = pair[0]

        new_keypints = []
        joint_split = keypoints.unbind(1)
        for i, _ in enumerate(joint_split):
            new_keypints.append(joint_split[flip_lut[i]])
        new_keypints = torch.stack(new_keypints, dim=1)
        return new_keypints

    def _prepare_keypoints(self, keypoints):
        labels = []
        joints = []
        for box in range(keypoints.shape[0]):
            # box_joints = []
            # box_labels = []
            for joint in range(keypoints.shape[1]):
                if keypoints[box, joint, 2] >= self.min_keypoint_visible:
                    labels.append(torch.as_tensor(joint))
                    joints.append(torch.as_tensor(keypoints[box, joint, :2]))

            # labels.append(box_labels)
            # joints.append(box_joints)
        if len(joints) == 0:
            return {"labels": torch.zeros([0], dtype=torch.int64), "joints": torch.zeros([0, 2], dtype=torch.float)}
        return {"labels": torch.stack(labels, dim=0), "joints": torch.stack(joints, dim=0)}

    def _prepare_and_filter(self, image, target):
        if self.min_num_keypoints is not None and self.min_num_keypoints > 0:
            if "keypoints" not in target:
                None
            # print(box_target["keypoints"][..., 2] > 1)
            num_keypoints = torch.sum(target["keypoints"][..., 2] >= self.min_keypoint_visible)
            if num_keypoints < self.min_num_keypoints:
                None
            # print(f"{box_target['keypoints']} {num_keypoints}")
        if target["flipped"]:
            target["keypoints"] = self.hflip_annotations(target["keypoints"])

        target.update(self._prepare_keypoints(target["keypoints"]))
        return image, target

    def __iter__(self):

        ids = copy.deepcopy(self.ids)
        for splitter in self.splitters:
            ids = splitter(ids)

        if self.shuffle:
            self.random_gen.shuffle(ids)
            # logging.info(f"Dataset on rank {rank}: {keys[:3]}")

        for id in ids:
            try:
                image = self._load_image(id)
                target = self._load_target(id)
            except:
                logging.warning(f"Image not found {id}")
                continue

            target = {"image_id": id, "annotations": target}

            image, target = self.prepare(image, target)
            # target.update({"image_id": id})
            # print(target)
            # exit()
            if "keypoints" not in target:
                continue
            for i, (box, keypoints, area, label, iscrowd) in enumerate(
                zip(
                    target["boxes"].unbind(0),
                    target["keypoints"].unbind(0),
                    target["area"].unbind(0),
                    target["labels"].unbind(0),
                    target["iscrowd"].unbind(0),
                )
            ):
                # print("#####")
                # print(box)
                # print(keypoints)
                # print(area)
                # skip transformation if there is already nothing

                box_target = {
                    **target,
                    "original_boxes": torch.unsqueeze(box, dim=0),
                    "boxes": torch.unsqueeze(box, dim=0),
                    "area": torch.unsqueeze(area, dim=0),
                    "keypoints": torch.unsqueeze(keypoints, dim=0),
                    "labels": torch.unsqueeze(label, dim=0),
                    "iscrowd": torch.unsqueeze(iscrowd, dim=0),
                    "flipped": False,
                }

                ar_box = boxes_aspect_ratio(box, self.ar)
                scaled_box = boxes_scale(ar_box, self.scale_boxes, size=target["size"])
                scaled_box_cxcywh = box_xyxy_to_cxcywh(scaled_box)
                scaled_box = scaled_box.numpy().tolist()
                scaled_box_wh = scaled_box_cxcywh.numpy().tolist()
                region = [scaled_box[1], scaled_box[0], scaled_box_wh[3], scaled_box_wh[2]]

                # print(f"{scaled_box} {ar_box} {box} {image.size} {region}")

                box_image, box_target = crop(
                    image,
                    box_target,
                    region,
                )

                # print(f"{box} {box_cxcywh} {box_image.size} {box_image2.size}")
                result = {}
                if self.transforms is not None:
                    box_image, box_target = self.transforms(box_image, box_target)
                    filtered = self._prepare_and_filter(box_image, box_target)
                    if filtered is None:
                        continue
                    result.update({"image": filtered[0], "target": filtered[1]})

                if self.weak_transforms is not None:
                    weak_box_image, weak_box_target = self.weak_transforms(box_image, box_target)
                    filtered = self._prepare_and_filter(weak_box_image, weak_box_target)
                    if filtered is None:
                        continue
                    result.update({"weak_image": filtered[0], "weak_target": filtered[1]})

                if self.strong_transforms is not None:

                    strong_box_image, strong_box_target = self.strong_transforms(box_image, box_target)
                    filtered = self._prepare_and_filter(strong_box_image, strong_box_target)
                    if filtered is None:
                        continue
                    result.update({"strong_image": filtered[0], "strong_target": filtered[1]})

                yield result


class CocoPoseIteratorDatasetPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super(CocoPoseIteratorDatasetPipeline, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def call(self, datasets=None, **kwargs):
        return CocoPoseIteratorDataset(*self.args, **self.kwargs, **kwargs)


def make_pose_transforms(image_set, type="weak"):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    mean = [int(255 * x) for x in [0.485, 0.456, 0.406]]

    scales = [192, 224, 256, 288, 320, 352, 384]

    if image_set == "train":
        if type == "normal":
            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomResize(scales, max_size=384),
                    T.PadCropToSize(size=[288, 384], fill=mean),
                    normalize,
                ]
            )
        elif type == "weak":
            return T.Compose(
                [
                    # T.RandomHorizontalFlip(),
                    T.RandomResize(scales, max_size=384),
                    T.PadCropToSize(size=[288, 384], fill=mean),
                    normalize,
                ]
            )
        elif type == "strong":
            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        [
                            T.Identity(),
                            T.AutoContrast(),
                            T.RandomEqualize(),
                            T.RandomSolarize(),
                            T.RandomColor(),
                            T.RandomContrast(),
                            T.RandomBrightness(),
                            T.RandomSharpness(),
                            T.RandomPosterize(),
                        ]
                    ),
                    T.RandomSelect(
                        [
                            T.RandomAffine(x=(-0.1, 0.1), fill=mean),
                            T.RandomAffine(y=(-0.1, 0.1), fill=mean),
                            T.RandomAffine(angle=(-30, 30), fill=mean),
                            T.Compose(
                                [
                                    T.RandomAffine(sx=(-30, 30), fill=mean),
                                    T.RandomAffine(sy=(-30, 30), fill=mean),
                                ]
                            ),
                        ],
                    ),
                    T.RandomResize(scales, max_size=384),
                    T.PadCropToSize(size=[288, 384], fill=mean),
                    normalize,
                ]
            )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([288], max_size=384),
                T.PadCropToSize(size=[288, 384], fill=mean),
                normalize,
            ]
        )
    if image_set == "infer":
        return T.Compose(
            [
                T.RandomResize([288], max_size=384),
                T.PadCropToSize(size=[288, 384], fill=mean),
                normalize,
            ]
        )
    raise ValueError(f"unknown {image_set}")


@DatasetsManager.export("coco_pose_iterator")
class CocoPoseIteratorDataloader(CocoIteratorDataloader):
    def __init__(self, args=None, **kwargs) -> None:
        super(CocoPoseIteratorDataloader, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.scale_boxes = dict_args.get("scale_boxes")
        self.min_keypoint_visible = dict_args.get("min_keypoint_visible")

    def train(self):

        pipelines = []
        for image, annotation in zip(self.train_path, self.train_annotation_path):
            pipeline = CocoPoseIteratorDatasetPipeline(
                image,
                annotation,
                transforms=make_pose_transforms("train", "normal"),
                return_masks=self.mask,
                shuffle=True,
                label_whitelist=self.label_whitelist,
                min_area=self.min_area,
                min_num_keypoints=self.min_num_keypoints,
                scale_boxes=self.scale_boxes,
                min_keypoint_visible=self.min_keypoint_visible,
            )

            pipelines.append(pipeline)

        pipeline = SequencePipeline([ConcatShufflePipeline(pipelines), RepeatPipeline()])
        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )
        return dataloader

    def val(self):
        pipeline = CocoPoseIteratorDatasetPipeline(
            self.val_path,
            self.val_annotation_path,
            transforms=make_pose_transforms("val"),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
            scale_boxes=self.scale_boxes,
            shuffle=False,
            min_keypoint_visible=self.min_keypoint_visible,
        )

        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )
        return dataloader

    def test(self):
        pipeline = CocoPoseIteratorDatasetPipeline(
            self.test_path,
            self.test_annotation_path,
            transforms=make_pose_transforms("val"),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
            scale_boxes=self.scale_boxes,
            shuffle=False,
            min_keypoint_visible=self.min_keypoint_visible,
        )
        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )
        return dataloader

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser = super(CocoPoseIteratorDataloader, cls).add_args(parser)

        parser.add_argument("--scale_boxes", type=float, default=1.0)
        parser.add_argument("--min_keypoint_visible", type=float, default=1.0)

        return parser
