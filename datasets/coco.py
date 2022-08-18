import argparse
import logging
import random
import copy
import numpy as np

from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List

from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import utils.misc as utils

import datasets.transforms as T

from datasets.datasets import DatasetsManager

from torchvision.datasets import VisionDataset


def get_node_rank():
    # if not torch.distributed.is_initialized():

    node_rank = os.environ.get("LOCAL_RANK")
    if node_rank is not None:
        return node_rank

    return 0


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {"weak_image": image, "weak_target": target}

    def __len__(self) -> int:
        return len(self.ids)


class CocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, label_whitelist=None, min_area=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.label_whitelist = label_whitelist
        self.min_area = min_area

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)

        if self.label_whitelist is not None:
            target = [x for x in target if x["category_id"] in self.label_whitelist]

        if self.min_area is not None:
            target = [x for x in target if x["area"] > self.min_area]

        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return {"weak_image": img, "weak_target": target}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, type="weak", train_size=None, val_size=None, infer_size=None, max_size=None):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    mean = [int(255 * x) for x in [0.485, 0.456, 0.406]]

    if max_size is None:
        max_size = 1333

    if val_size is None:
        val_size = 800

    if infer_size is None:
        infer_size = 800

    if train_size is None:
        train_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        if type == "normal":
            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        [
                            T.RandomResize(train_size, max_size=max_size),
                            T.Compose(
                                [
                                    T.RandomResize([400, 500, 600]),
                                    T.RandomSizeCrop(384, 600),
                                    T.RandomResize(train_size, max_size=max_size),
                                ]
                            ),
                        ]
                    ),
                    normalize,
                ]
            )
        elif type == "weak":
            return T.Compose(
                [
                    # T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        [
                            T.RandomResize(train_size, max_size=max_size),
                            # T.Compose(
                            #     [
                            #         T.RandomResize([400, 500, 600]),
                            #         T.RandomSizeCrop(384, 600),
                            #         T.RandomResize(train_size, max_size=max_size),
                            #     ]
                            # ),
                        ]
                    ),
                    normalize,
                ]
            )
        elif type == "strong":
            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        [
                            T.RandomResize(train_size, max_size=max_size),
                            T.Compose(
                                [
                                    T.RandomResize([400, 500, 600]),
                                    T.RandomSizeCrop(384, 600),
                                    T.RandomResize(train_size, max_size=max_size),
                                ]
                            ),
                        ]
                    ),
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
                    normalize,
                    T.RandomErasing(),
                ]
            )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([val_size], max_size=max_size),
                normalize,
            ]
        )

    if image_set == "infer":
        return T.Compose(
            [
                T.RandomResize([infer_size], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


from datasets.datasets import DatasetsManager
from datasets.pipeline import (
    MapPipeline,
    Pipeline,
    Dataset,
    MapDataset,
    MsgPackPipeline,
    SequencePipeline,
    ConcatShufflePipeline,
    ConcatPipeline,
    DummyPipeline,
    ImagePipeline,
    split_chunk_by_nodes,
    split_chunk_by_workers,
)


@DatasetsManager.export("coco")
class CocoDataloader:
    def __init__(self, args=None, **kwargs):
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.mask = dict_args.get("mask", None)

        self.train_path = dict_args.get("train_path", None)
        self.train_annotation_path = dict_args.get("train_annotation_path", None)

        self.train_random_sizes = dict_args.get("train_random_sizes", None)
        self.max_size = dict_args.get("max_size", 800)

        self.val_path = dict_args.get("val_path", None)
        self.val_annotation_path = dict_args.get("val_annotation_path", None)

        self.test_path = dict_args.get("test_path", None)
        self.test_annotation_path = dict_args.get("test_annotation_path", None)

        self.infer_path = dict_args.get("infer_path", None)
        self.infer_size = dict_args.get("infer_size", None)

        self.batch_size = dict_args.get("batch_size", None)
        self.num_workers = dict_args.get("num_workers", None)

        self.label_whitelist = dict_args.get("label_whitelist", None)
        self.min_area = dict_args.get("min_area", None)

    def train(self):

        dataset = CocoDetection(
            self.train_path,
            self.train_annotation_path,
            transforms=make_coco_transforms("train"),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=utils.collate_fn,
            sampler=torch.utils.data.RandomSampler(dataset),
        )
        return dataloader

    def val(self):

        dataset = CocoDetection(
            self.val_path,
            self.val_annotation_path,
            transforms=make_coco_transforms("val"),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=utils.collate_fn,
        )
        return dataloader

    def test(self):
        dataset = CocoDetection(
            self.test_path,
            self.test_annotation_path,
            transforms=make_coco_transforms("val"),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=utils.collate_fn,
        )
        return dataloader

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--use_center_crop", action="store_true", help="verbose output")

        parser.add_argument("--train_path", type=str)
        parser.add_argument("--train_annotation_path", type=str)

        parser.add_argument("--train_random_sizes", type=int, nargs="+", default=[480, 512, 544, 576, 608, 640])
        parser.add_argument("--max_size", type=int, default=800)

        parser.add_argument("--val_path", type=str)
        parser.add_argument("--val_annotation_path", type=str)

        parser.add_argument("--test_path", type=str)
        parser.add_argument("--test_annotation_path", type=str)

        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=8)

        parser.add_argument("--label_whitelist", type=int, nargs="+")
        parser.add_argument("--min_area", type=float)

        return parser
