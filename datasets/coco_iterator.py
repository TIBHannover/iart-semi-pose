import logging
import argparse
import random
import copy
import os
import json
from typing import Any, Callable, Optional, Sequence, Tuple, List, Dict

import torch
from PIL import Image
import utils.misc as utils

import datasets.transforms as T
from datasets.datasets import DatasetsManager
from datasets.coco import make_coco_transforms, get_node_rank
from datasets.coco import ConvertCocoPolysToMask

from datasets.datasets import DatasetsManager
from datasets.pipeline import (
    Pipeline,
    Dataset,
    SequencePipeline,
    RepeatPipeline,
    ConcatShufflePipeline,
    split_chunk_by_nodes,
    split_chunk_by_workers,
)


class CocoIteratorDataset(Dataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        weak_transforms: Optional[Callable] = None,
        strong_transforms: Optional[Callable] = None,
        return_masks: bool = False,
        shuffle: bool = True,
        label_whitelist: Optional[List] = None,
        min_area: Optional[float] = None,
        min_num_keypoints: Optional[int] = None,
        seed: Optional[int] = 42,
        map_classes: Optional[Dict] = 42,
    ):
        super(CocoIteratorDataset, self).__init__()
        from pycocotools.coco import COCO

        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.target_transform = target_transform

        self.transforms = transforms
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms

        self.splitters = [split_chunk_by_workers]

        self.random_gen = random.Random(seed + get_node_rank())
        self.shuffle = shuffle

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.label_whitelist = label_whitelist
        self.min_area = min_area
        self.min_num_keypoints = min_num_keypoints
        self.map_classes = map_classes

    def _load_image(self, id: int) -> Image.Image:
        # print(self.coco.loadImgs(id))
        path = self.coco.loadImgs(id)[0]["file_name"]
        # print(path)
        # exit()
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _prepare_and_filter(self, image, target):
        if self.min_num_keypoints is not None and self.min_num_keypoints > 0:
            if "keypoints" not in target:
                return None
        # this is for detection only so we will delete keypoints here
        if "keypoints" in target:
            del target["keypoints"]
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

            if self.map_classes:
                new_target = []
                for x in target:
                    if str(x["category_id"]) in self.map_classes:
                        # print('TTTT')
                        x["category_id"] = self.map_classes[str(x["category_id"])]
                    new_target.append(x)

                # print(new_target)

                target = new_target

            if self.label_whitelist is not None:
                target = [x for x in target if x["category_id"] in self.label_whitelist]

            if self.min_area is not None:
                target = [x for x in target if x["area"] > self.min_area]

            target = {"image_id": id, "annotations": target}

            image, target = self.prepare(image, target)

            result = {}
            if self.transforms is not None:
                image, target = self.transforms(image, target)
                filtered = self._prepare_and_filter(image, target)
                if filtered is None:
                    continue
                result.update({"image": filtered[0], "target": filtered[1]})

            if self.weak_transforms is not None:
                weak_image, weak_target = self.weak_transforms(image, target)
                filtered = self._prepare_and_filter(weak_image, weak_target)
                if filtered is None:
                    continue
                result.update({"weak_image": filtered[0], "weak_target": filtered[1]})

            if self.strong_transforms is not None:
                strong_image, strong_target = self.strong_transforms(image, target)
                filtered = self._prepare_and_filter(strong_image, strong_target)
                if filtered is None:
                    continue
                result.update({"strong_image": filtered[0], "strong_target": filtered[1]})
            # print(result["target"])
            yield result


class CocoIteratorDatasetPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super(CocoIteratorDatasetPipeline, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def call(self, datasets=None, **kwargs):
        return CocoIteratorDataset(*self.args, **self.kwargs, **kwargs)


@DatasetsManager.export("coco_iterator")
class CocoIteratorDataloader:
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
        self.max_size = dict_args.get("max_size", None)

        self.val_path = dict_args.get("val_path", None)
        self.val_annotation_path = dict_args.get("val_annotation_path", None)
        self.val_size = dict_args.get("val_size", None)

        self.test_path = dict_args.get("test_path", None)
        self.test_annotation_path = dict_args.get("test_annotation_path", None)

        self.infer_path = dict_args.get("infer_path", None)
        self.infer_annotation_path = dict_args.get("infer_annotation_path", None)
        self.infer_size = dict_args.get("infer_size", None)

        self.batch_size = dict_args.get("batch_size", None)
        self.train_map_classes = dict_args.get("train_map_classes", None)

        if self.train_map_classes:
            self.train_map_classes = [json.loads(x) for x in self.train_map_classes]
        elif isinstance(self.train_annotation_path, (set, list)):
            self.train_map_classes = [None for x in self.train_annotation_path]

        self.val_map_classes = dict_args.get("val_map_classes", None)
        if self.val_map_classes:
            self.val_map_classes = json.loads(self.val_map_classes)

        self.test_map_classes = dict_args.get("test_map_classes", None)
        if self.test_map_classes:
            self.test_map_classes = json.loads(self.test_map_classes)

        self.num_workers = dict_args.get("num_workers", None)

        self.label_whitelist = dict_args.get("label_whitelist", None)
        self.min_area = dict_args.get("min_area", None)
        self.min_num_keypoints = dict_args.get("min_num_keypoints", None)
        self.seed = dict_args.get("seed")

    def train(self):
        pipelines = []
        for image, annotation, map_classes in zip(self.train_path, self.train_annotation_path, self.train_map_classes):
            pipeline = CocoIteratorDatasetPipeline(
                image,
                annotation,
                transforms=make_coco_transforms(
                    "train", "normal", train_size=self.train_random_sizes, max_size=self.max_size
                ),
                # strong_transforms=make_coco_transforms("train", "strong"),
                return_masks=self.mask,
                shuffle=True,
                label_whitelist=self.label_whitelist,
                min_area=self.min_area,
                min_num_keypoints=self.min_num_keypoints,
                map_classes=map_classes,
            )
            pipelines.append(pipeline)
        pipeline = ConcatShufflePipeline(pipelines)

        pipeline = SequencePipeline([pipeline, RepeatPipeline()])
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

        pipeline = CocoIteratorDatasetPipeline(
            self.val_path,
            self.val_annotation_path,
            transforms=make_coco_transforms("val", val_size=self.val_size, max_size=self.max_size),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
            map_classes=self.val_map_classes,
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
        pipeline = CocoIteratorDatasetPipeline(
            self.test_path,
            self.test_annotation_path,
            transforms=make_coco_transforms("val", val_size=self.val_size, max_size=self.max_size),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
            map_classes=self.test_map_classes,
        )
        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )
        return dataloader

    def infer(self):
        pipeline = CocoIteratorDatasetPipeline(
            self.infer_path,
            self.infer_annotation_path,
            transforms=make_coco_transforms("infer", infer_size=self.infer_size, max_size=self.max_size),
            return_masks=self.mask,
            label_whitelist=self.label_whitelist,
            min_area=self.min_area,
            map_classes=None,
        )
        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )
        return dataloader

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--use_center_crop", action="store_true", help="verbose output")

        parser.add_argument("--train_path", type=str, nargs="+")
        parser.add_argument("--train_annotation_path", type=str, nargs="+")
        parser.add_argument("--train_map_classes", type=str, nargs="+")

        parser.add_argument(
            "--train_random_sizes", type=int, nargs="+", default=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        )
        parser.add_argument("--max_size", type=int, default=1333)

        parser.add_argument("--val_path", type=str)
        parser.add_argument("--val_annotation_path", type=str)
        parser.add_argument("--val_size", type=int, default=800)
        parser.add_argument("--val_map_classes", type=str)

        parser.add_argument("--test_path", type=str)
        parser.add_argument("--test_annotation_path", type=str)
        parser.add_argument("--test_map_classes", type=str)

        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=2)

        parser.add_argument("--label_whitelist", type=int, nargs="+")
        parser.add_argument("--min_area", type=float)
        parser.add_argument("--min_num_keypoints", type=int)

        parser.add_argument("--seed", type=int, default=42)

        parser.add_argument("--infer_path", type=str)
        parser.add_argument("--infer_size", type=int, default=800)
        parser.add_argument("--infer_annotation_path", type=str)
        return parser
