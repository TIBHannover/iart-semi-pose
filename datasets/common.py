import argparse
import json
import os
import random
import logging
import copy
from re import X
from typing import Any, Callable, Optional, Tuple, List, Dict

import torch
from PIL import Image

from datasets.coco import get_node_rank
from datasets.pipeline import MergePipeline, Pipeline, Dataset, split_chunk_by_workers

from datasets.transforms import crop
from utils.box_ops import box_xyxy_to_cxcywh, boxes_scale, boxes_aspect_ratio, box_iou


class IteratorDataset(Dataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Optional[Callable] = None,
        weak_transforms: Optional[Callable] = None,
        strong_transforms: Optional[Callable] = None,
        crop_boxes: bool = False,
        scale_boxes: float = None,
        shuffle: bool = True,
        min_area: float = None,
        box_max_iou: float = None,
        min_score: float = None,
        seed: int = 42,
        image_filter: str = None,
    ):
        super(IteratorDataset, self).__init__()

        self.root = root
        self.annFile = annFile
        self.samples = []

        self.transforms = transforms
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms

        self.scale_boxes = scale_boxes
        if scale_boxes is None:
            self.scale_boxes = 1.0

        self.target_height = 384
        self.target_width = 288
        self.ar = self.target_width / self.target_height

        self.crop_boxes = crop_boxes

        self.splitters = [split_chunk_by_workers]

        self.random_gen = random.Random(seed + get_node_rank())
        self.shuffle = shuffle
        self.load_annotations()

        self.min_area = min_area
        self.box_max_iou = box_max_iou
        self.min_score = min_score
        self.image_filter = image_filter

        if self.image_filter is not None:
            filter_hashs = set()
            with open(self.image_filter) as f:
                for line in f:
                    data = json.loads(line)
                    filter_hashs.add(data["hash"])
            new_samples = []
            for x in self.samples:
                if x["hash"] in filter_hashs:
                    new_samples.append(x)
            self.samples = new_samples

        logging.info(f"IteratorDataset::Found {len(self.samples)}")

    def load_annotations(self):
        self.samples = []
        with open(self.annFile) as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)

    def _load_image(self, sample: Dict) -> Image.Image:
        return Image.open(os.path.join(self.root, sample.get("path"))).convert("RGB")

    def _prepare_sample(self, image, target):
        target = copy.deepcopy(target)
        result = {}
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            result.update({"image": image, "target": target})

        if self.weak_transforms is not None:
            weak_image, weak_target = self.weak_transforms(image, target)

            result.update({"weak_image": weak_image, "weak_target": weak_target})

        if self.strong_transforms is not None:
            strong_image, strong_target = self.strong_transforms(image, target)
            result.update({"strong_image": strong_image, "strong_target": strong_target})
        return result

    def __iter__(self):

        samples = copy.deepcopy(self.samples)
        for splitter in self.splitters:
            samples = splitter(samples)

        if self.shuffle:
            self.random_gen.shuffle(samples)
            # logging.info(f"Dataset on rank {rank}: {keys[:3]}")

        for sample in samples:

            try:
                image = self._load_image(sample)
            except:
                logging.warning(f"Image not found {sample}")
                continue
            target = {}

            target = {"image_id": sample.get("id")}

            w, h = image.size
            target["size"] = torch.tensor([h, w])
            target["origin_size"] = torch.tensor([h, w])
            if self.crop_boxes:
                boxes = sample.get("boxes")
                scores = sample.get("scores")

                if boxes is None or len(boxes) == 0:
                    continue

                if scores is None:
                    scores = [1.0 for x in boxes]

                delete_boxes = None
                if self.box_max_iou:
                    # print(torch.as_tensor(boxes))
                    iou, _ = box_iou(torch.as_tensor(boxes), torch.as_tensor(boxes))
                    diag = torch.diagonal(iou)
                    diag[:] = 0
                    # print(f"iou {iou}")
                    max_iou = torch.max(iou, 0).values
                    # print(f"max_iou {max_iou}")
                    delete_boxes = max_iou > self.box_max_iou
                    # print(f"delete_boxes {delete_boxes}")

                for i, (box, score) in enumerate(zip(boxes, scores)):
                    if delete_boxes is not None:
                        if delete_boxes[i]:
                            # print("BOX iou to large")
                            continue
                    if self.min_score:
                        if score < self.min_score:
                            # print(f"BOX score to small {score}")
                            continue
                    box = torch.squeeze(torch.as_tensor(box))
                    box_target = {**target, "boxes": torch.unsqueeze(box, dim=0), "flipped": False, "boxes_id": [i]}

                    ar_box = boxes_aspect_ratio(box, self.ar)
                    scaled_box = boxes_scale(ar_box, self.scale_boxes, size=target["size"])
                    scaled_box_cxcywh = box_xyxy_to_cxcywh(scaled_box)
                    scaled_box = scaled_box.numpy().tolist()
                    scaled_box_wh = scaled_box_cxcywh.numpy().tolist()
                    # print(f"## {self.min_area} {scaled_box_wh[3] * scaled_box_wh[2]}")
                    if self.min_area:
                        if scaled_box_wh[3] * scaled_box_wh[2] < self.min_area:
                            # print("BOX to small")
                            continue
                    region = [scaled_box[1], scaled_box[0], scaled_box_wh[3], scaled_box_wh[2]]

                    # print(f"{scaled_box} {ar_box} {box} {image.size} {region}")

                    box_image, box_target = crop(
                        image,
                        box_target,
                        region,
                    )
                    yield self._prepare_sample(box_image, box_target)
            else:
                yield self._prepare_sample(image, target)


class IteratorDatasetPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super(IteratorDatasetPipeline, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def call(self, datasets=None, **kwargs):
        return IteratorDataset(*self.args, **self.kwargs, **kwargs)
