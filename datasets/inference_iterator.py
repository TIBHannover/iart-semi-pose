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
from datasets.coco_iterator import CocoIteratorDataloader
from datasets.coco_iterator import CocoIteratorDatasetPipeline
from datasets.coco import make_coco_transforms, get_node_rank
from datasets.pipeline import MergePipeline, Pipeline, Dataset, split_chunk_by_workers
from datasets.common import IteratorDatasetPipeline
from datasets.coco_pose_iterator import make_pose_transforms


@DatasetsManager.export("boxes_inference_iterator")
class BoxesInferIteratorDataloader(CocoIteratorDataloader):
    def __init__(self, args=None, **kwargs) -> None:
        super(BoxesInferIteratorDataloader, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.infer_path = dict_args.get("infer_path", None)

        self.infer_size = dict_args.get("infer_size", None)
        self.infer_max_size = dict_args.get("infer_max_size", None)
        self.infer_annotation_path = dict_args.get("infer_annotation_path", None)

    def infer(self):
        pipeline = IteratorDatasetPipeline(
            self.infer_path,
            self.infer_annotation_path,
            transforms=make_coco_transforms("infer", infer_size=self.infer_size, max_size=self.infer_max_size),
            shuffle=False,
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
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser = super(BoxesInferIteratorDataloader, cls).add_args(parser)
        parser.add_argument("--infer_path", type=str)
        parser.add_argument("--infer_size", type=int, default=512)
        parser.add_argument("--infer_max_size", type=int, default=512)
        parser.add_argument("--infer_annotation_path", type=str)

        return parser


@DatasetsManager.export("pose_inference_iterator")
class PoseInferIteratorDataloader(CocoIteratorDataloader):
    def __init__(self, args=None, **kwargs) -> None:
        super(PoseInferIteratorDataloader, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.infer_path = dict_args.get("infer_path", None)

        self.infer_size = dict_args.get("infer_size", None)
        self.infer_max_size = dict_args.get("infer_max_size", None)
        self.infer_annotation_path = dict_args.get("infer_annotation_path", None)
        self.scale_boxes = dict_args.get("scale_boxes", None)
        self.image_filter = dict_args.get("image_filter", None)

    def infer(self):
        pipeline = IteratorDatasetPipeline(
            self.infer_path,
            self.infer_annotation_path,
            transforms=make_pose_transforms("infer", "weak"),
            crop_boxes=True,
            scale_boxes=self.scale_boxes,
            image_filter=self.image_filter,
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
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser = super(PoseInferIteratorDataloader, cls).add_args(parser)
        parser.add_argument("--infer_path", type=str)
        parser.add_argument("--infer_size", type=int, default=800)
        parser.add_argument("--infer_max_size", type=int, default=1333)
        parser.add_argument("--infer_annotation_path", type=str)
        parser.add_argument("--scale_boxes", type=float, default=1.0)
        parser.add_argument("--image_filter")

        return parser
