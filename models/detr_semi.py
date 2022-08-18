import logging
import argparse
import copy

import torch
from torch.nn import parameter

from models import ModelsManager
from models.detr import DETRModel
from .losses import BoxesSetCriterion
from .matcher import build_boxes_matcher
import utils.misc as utils
from utils import box_ops

from models.utils import EMA


from utils.box_ops import (
    boxes_transformation,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    boxes_to_abs,
    boxes_to_rel,
    boxes_fit_size,
)


def box_area(boxes_xyxy):
    return torch.abs(boxes_xyxy[..., 0] - boxes_xyxy[..., 2]) * torch.abs(boxes_xyxy[..., 1] - boxes_xyxy[..., 3])


@ModelsManager.export("detr_semi")
class DETRSemiModel(DETRModel):
    def __init__(self, args=None, **kwargs):
        super(DETRSemiModel, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.fixmatch_threshold = dict_args.get("fixmatch_threshold", 0.9)
        self.fixmatch_background_threshold = dict_args.get("fixmatch_background_threshold")
        self.num_top_negative = dict_args.get("num_top_negative")
        self.weight_unsupervised = dict_args.get("weight_unsupervised", 1.0)
        self.use_negative_boxes = dict_args.get("use_negative_boxes")
        self.sample_negative_boxes = dict_args.get("sample_negative_boxes")
        self.min_area_strong = dict_args.get("min_area_strong")

        self.teacher = EMA(self.model)
        self.boxes_images = []

        num_classes = 91

        matcher = build_boxes_matcher(args)
        weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
        weight_dict["loss_giou"] = args.giou_loss_coef
        if args.masks:
            weight_dict["loss_mask"] = args.mask_loss_coef
            weight_dict["loss_dice"] = args.dice_loss_coef
        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "cardinality"]
        if args.masks:
            losses += ["masks"]
        self.criterion = BoxesSetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=args.eos_coef,
            losses=losses,
            negativ_threshold=self.fixmatch_background_threshold,
            num_top_negative=self.num_top_negative,
        )

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.teacher:
            self.teacher.update(self.model)

    def _get_fixmatch_prediction(self, weak_outputs):
        predictions = {"boxes": [], "labels": [], "scores": []}
        batch_size = weak_outputs["pred_logits"].shape[0]

        label_softmax = torch.softmax(weak_outputs["pred_logits"], dim=-1)
        top_prediction = label_softmax > self.fixmatch_threshold
        boxes_pos = top_prediction[..., :-1].nonzero()

        for b in range(batch_size):
            boxes = []
            labels = []

            boxes_sample = boxes_pos[boxes_pos[:, 0] == b]
            for box in boxes_sample.unbind(0):
                box_index = box[1]
                box_cls = box[2]
                box_cxcywh = weak_outputs["pred_boxes"][b][box_index]
                if self.label_whitelist is None:
                    labels.append(box_cls)
                    boxes.append(box_cxcywh)
                else:
                    box_label = box_cls.detach().cpu().numpy().item()
                    if box_label in self.label_whitelist:

                        # print(box_label)
                        labels.append(box_cls)
                        boxes.append(box_cxcywh)

                    # exit()
            #     print(f"{box} {box_index} {box_cls} {box_cxcywh}")
            # print(boxes)
            # print(labels)
            if len(boxes) > 0:
                predictions["boxes"].append(torch.stack(boxes, dim=0))
                predictions["labels"].append(torch.stack(labels, dim=0))
            else:
                predictions["boxes"].append(torch.zeros([0, 4], device=label_softmax.device))
                predictions["labels"].append(torch.zeros([0], dtype=torch.int64, device=label_softmax.device))

        return predictions

    def _build_unsupervised_target(self, weak_outputs, weak_targets, strong_targets):

        unsupervised_target = {"boxes": [], "labels": [], "neg_boxes": [], "size": strong_targets["size"]}
        # print(supervised["target"])
        # exit()

        batch_size = weak_outputs["pred_logits"].shape[0]
        # background_label = weak_outputs["pred_logits"].shape[-1] - 1
        # print(f"Background {background_label}")

        # print(weak_outputs)
        # print(weak_outputs['pred_logits'].shape)
        label_softmax = torch.softmax(weak_outputs["pred_logits"], dim=-1)
        top_prediction = label_softmax > self.fixmatch_threshold
        boxes_pos = top_prediction[..., :-1].nonzero()

        # boxes_neg = top_prediction[..., -1].nonzero()

        # exit()
        # transform_cordinates
        for b in range(batch_size):
            boxes = []
            # neg_boxes = []
            labels = []
            inv_transformation = torch.linalg.inv(weak_targets["transformation"][b])
            weak_boxes_abs = boxes_to_abs(
                box_cxcywh_to_xyxy(weak_outputs["pred_boxes"][b]), size=weak_targets["size"][b]
            )
            boxes_origins_abs = boxes_transformation(weak_boxes_abs, inv_transformation)

            strong_boxes_transformation_abs = boxes_transformation(
                boxes_origins_abs, strong_targets["transformation"][b]
            )

            strong_boxes_transformation_abs = boxes_fit_size(
                strong_boxes_transformation_abs, size=strong_targets["size"][b]
            )

            strong_boxes_transformation = box_xyxy_to_cxcywh(
                boxes_to_rel(strong_boxes_transformation_abs, size=strong_targets["size"][b])
            )

            boxes_sample = boxes_pos[boxes_pos[:, 0] == b]
            # neg_boxes_sample = boxes_neg[boxes_neg[:, 0] == b]

            # we sample some hard negativ prediction
            # for box in neg_boxes_sample.unbind(0):
            #     box_index = box[1]
            #     box_cxcywh = strong_boxes_transformation[box_index]
            #     neg_boxes.append(box_cxcywh)

            for box in boxes_sample.unbind(0):
                box_index = box[1]
                box_cls = box[2]
                box_cxcywh = strong_boxes_transformation[box_index]
                area = box_area(strong_boxes_transformation_abs[box_index])

                if area < self.min_area_strong:
                    continue

                if self.label_whitelist is None:
                    labels.append(box_cls)
                    boxes.append(box_cxcywh)
                else:
                    box_label = box_cls.detach().cpu().numpy().item()
                    if box_label in self.label_whitelist:

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

            # if len(neg_boxes) > 0:
            #     unsupervised_target["neg_boxes"].append(torch.stack(neg_boxes, dim=0))
            # else:
            #     unsupervised_target["neg_boxes"].append(torch.zeros([0, 4], device=label_softmax.device))

        return unsupervised_target

    def training_step(self, batch, batch_idx):
        supervised = batch["supervised"]
        samples, targets = supervised["image"], supervised["target"]
        # print(targets)
        # print(targets["boxes"])
        # exit()
        unsupervised = batch["unsupervised"]
        weak_samples, weak_targets, strong_samples, strong_targets = (
            unsupervised["weak_image"],
            unsupervised["weak_target"],
            unsupervised["strong_image"],
            unsupervised["strong_target"],
        )

        # supervised part
        # print(f"NORMAL {batch_size} {utils.get_rank()} {self.trainer.global_step}")
        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)

        self.log("train/supervised/count_neg", loss_dict["count_neg"])
        self.log("train/supervised/count_pos", loss_dict["count_pos"])
        self.log("train/supervised/dist", loss_dict["count_neg"] / (loss_dict["count_pos"] + 1))
        self.log("train/supervised/loss_ce", loss_dict["loss_ce"])
        self.log("train/supervised/loss_bbox", loss_dict["loss_bbox"])
        self.log("train/supervised/cardinality_error", loss_dict["cardinality_error"])

        weight_dict = self.criterion.weight_dict
        supervised_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # unsupervised transformation
        with torch.no_grad():
            weak_outputs = self.teacher(weak_samples)
            unsupervised_target = self._build_unsupervised_target(weak_outputs, weak_targets, strong_targets)
        outputs = self.model(strong_samples)
        loss_dict = self.criterion(
            outputs,
            unsupervised_target,
            pos_only=not self.use_negative_boxes,
            sample_negative=self.sample_negative_boxes,
        )

        self.log("train/unsupervised/count_neg", loss_dict["count_neg"])
        self.log("train/unsupervised/count_pos", loss_dict["count_pos"])
        self.log("train/unsupervised/dist", loss_dict["count_neg"] / (loss_dict["count_pos"] + 1))
        self.log("train/unsupervised/loss_ce", loss_dict["loss_ce"])
        self.log("train/unsupervised/loss_bbox", loss_dict["loss_bbox"])
        self.log("train/unsupervised/cardinality_error", loss_dict["cardinality_error"])

        weight_dict = self.criterion.weight_dict
        unsupervised_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # for ploting the images
        self.boxes_images = utils.detach_all(
            [
                {
                    "ids": targets["image_id"],
                    "names": "supervised",
                    "images": samples.tensors,
                    "boxes": targets["boxes"],
                    "sizes": targets["size"],
                },
                {
                    "ids": weak_targets["image_id"],
                    "names": "unsupervised_weak",
                    "images": weak_samples.tensors,
                    "boxes": self._get_fixmatch_prediction(weak_outputs)["boxes"],
                    "sizes": weak_targets["size"],
                },
                {
                    "ids": strong_targets["image_id"],
                    "names": "unsupervised_strong",
                    "images": strong_samples.tensors,
                    "boxes": unsupervised_target["boxes"],
                    "sizes": strong_targets["size"],
                },
            ]
        )

        # exit()
        self.log("train/supervised_loss", supervised_losses)
        self.log("train/unsupervised_loss", self.weight_unsupervised * unsupervised_losses)
        self.log("train/loss", supervised_losses + self.weight_unsupervised * unsupervised_losses)

        return {"loss": supervised_losses + self.weight_unsupervised * unsupervised_losses}

    @classmethod
    def add_args(cls, parent_parser):
        print("DETRSemi::add_args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser = super(DETRSemiModel, cls).add_args(parser)

        parser.add_argument("--fixmatch_threshold", type=float, default=0.9)
        parser.add_argument("--fixmatch_background_threshold", type=float)
        parser.add_argument("--num_top_negative", type=int, default=5)
        parser.add_argument("--min_area_strong", type=float, default=1024)

        parser.add_argument("--use_negative_boxes", action="store_true")
        parser.add_argument("--sample_negative_boxes", action="store_true")
        parser.add_argument(
            "--weight_unsupervised", default=1, type=float, help="L1 box coefficient in the matching cost"
        )
        return parser

    @classmethod
    def tunning_scopes(cls):
        from ray import tune

        parameter_scope = {}

        if hasattr(super(DETRSemiModel, cls), "tunning_scopes"):
            parameter_scope.update(super(DETRSemiModel, cls).tunning_scopes())
        parameter_scope.update(
            {
                "weight_unsupervised": tune.choice([0.1, 0.5, 1.0, 2.0]),
                "fixmatch_threshold": tune.choice([0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
            }
        )
        return parameter_scope