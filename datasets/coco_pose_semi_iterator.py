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
from datasets.coco_pose_iterator import CocoPoseIteratorDataloader, make_pose_transforms
from datasets.coco_pose_iterator import CocoPoseIteratorDatasetPipeline
from datasets.coco_iterator import CocoIteratorDatasetPipeline
from datasets.coco_pose_iterator import make_coco_transforms, get_node_rank
from datasets.pipeline import (
    MergePipeline,
    SequencePipeline,
    CachePipeline,
    RepeatPipeline,
    ConcatShufflePipeline,
    split_chunk_by_workers,
)
from datasets.common import IteratorDatasetPipeline


@DatasetsManager.export("coco_pose_semi_iterator")
class CocoPoseSemiIteratorDataloader(CocoPoseIteratorDataloader):
    def __init__(self, args=None, **kwargs) -> None:
        super(CocoPoseSemiIteratorDataloader, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.unsupervised_path = dict_args.get("unsupervised_path", None)
        self.unsupervised_annotation_path = dict_args.get("unsupervised_annotation_path", None)

        self.unsupervised_min_area = dict_args.get("unsupervised_min_area")
        self.unsupervised_max_iou = dict_args.get("unsupervised_max_iou")
        self.unsupervised_min_score = dict_args.get("unsupervised_min_score")

    def train(self):

        pipelines = []
        for image, annotation in zip(self.train_path, self.train_annotation_path):
            pipeline = CocoPoseIteratorDatasetPipeline(
                image,
                annotation,
                transforms=make_pose_transforms("train", "normal"),
                shuffle=True,
                scale_boxes=self.scale_boxes,
            )
            pipelines.append(pipeline)
        supervised_pipeline = ConcatShufflePipeline(pipelines)

        unsupervised_pipeline = IteratorDatasetPipeline(
            self.unsupervised_path,
            self.unsupervised_annotation_path,
            weak_transforms=make_pose_transforms("train", "weak"),
            strong_transforms=make_pose_transforms("train", "strong"),
            shuffle=True,
            crop_boxes=True,
            scale_boxes=self.scale_boxes,
            min_area=self.unsupervised_min_area,
            box_max_iou=self.unsupervised_max_iou,
            min_score=self.unsupervised_min_score,
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
                CachePipeline(cache_size=256, shuffle=True),
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
        parser = super(CocoPoseSemiIteratorDataloader, cls).add_args(parser)
        parser.add_argument("--unsupervised_path", type=str)
        parser.add_argument("--unsupervised_annotation_path", type=str)
        parser.add_argument("--unsupervised_min_area", type=float)
        parser.add_argument("--unsupervised_max_iou", type=float)
        parser.add_argument("--unsupervised_min_score", type=float)

        return parser

    @classmethod
    def tunning_scopes(cls):
        from ray import tune

        parameter_scope = {}

        if hasattr(super(CocoPoseSemiIteratorDataloader, cls), "tunning_scopes"):
            parameter_scope.update(super(CocoPoseSemiIteratorDataloader, cls).tunning_scopes())
        parameter_scope.update(
            {
                "unsupervised_min_score": tune.choice([0.6, 0.7, 0.8, 0.9, 0.9]),
            }
        )
        return parameter_scope