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
from datasets.pipeline import (
    MergePipeline,
    RepeatPipeline,
    ConcatShufflePipeline,
    SequencePipeline,
    split_chunk_by_workers,
)
from datasets.common import IteratorDatasetPipeline


@DatasetsManager.export("coco_semi_iterator")
class CocoSemiIteratorDataloader(CocoIteratorDataloader):
    def __init__(self, args=None, **kwargs) -> None:
        super(CocoSemiIteratorDataloader, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.unsupervised_path = dict_args.get("unsupervised_path", None)
        self.unsupervised_annotation_path = dict_args.get("unsupervised_annotation_path", None)
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
                seed=self.seed,
                map_classes=map_classes,
            )
            pipelines.append(pipeline)
        supervised_pipeline = ConcatShufflePipeline(pipelines)

        unsupervised_pipeline = IteratorDatasetPipeline(
            self.unsupervised_path,
            self.unsupervised_annotation_path,
            weak_transforms=make_coco_transforms(
                "train", "weak", train_size=self.train_random_sizes, max_size=self.max_size
            ),
            strong_transforms=make_coco_transforms(
                "train", "strong", train_size=self.train_random_sizes, max_size=self.max_size
            ),
            shuffle=True,
            seed=self.seed,
        )

        def merge_fn(x):
            return {"supervised": x[0], "unsupervised": x[1]}

        pipeline = SequencePipeline(
            [
                MergePipeline(
                    pipelines=[supervised_pipeline, unsupervised_pipeline],
                    merge_fn=merge_fn,
                ),
                RepeatPipeline(),
            ]
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
        parser = super(CocoSemiIteratorDataloader, cls).add_args(parser)
        parser.add_argument("--unsupervised_path", type=str)
        parser.add_argument("--unsupervised_annotation_path", type=str)

        parser.add_argument("--seed", type=int, default=42)

        return parser
