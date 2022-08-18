import logging
import argparse
import copy
from unittest import result

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models import ModelsManager
from models.detr import DETRModel, MLP
from models.utils import EMA
import utils.misc as utils
from utils import box_ops


from utils.misc import NestedTensor, nested_tensor_from_tensor_list, unflat_dict, flat_dict

from utils.box_ops import points_transformation, point_to_abs, point_to_rel


from .pose_backbone import build_backbone
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from .transformer import build_transformer
from .matcher import build_coords_matcher
from .losses import CoordsSetCriterion

from datasets.coco_eval import CocoKeypointsEvaluator
from .pose_transformer import PoseTransformerModel


class COCOPoseFlipper:
    def __init__(self):
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

        self.flip_lut = {}
        for x in range(len(self.keypoints)):
            self.flip_lut[x] = x

        for pair in self.flip_pairs:
            self.flip_lut[pair[0]] = pair[1]
            self.flip_lut[pair[1]] = pair[0]

    def hflip_annotations(self, keypoints):
        print(keypoints)
        exit()
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
        return new_keypint

    def hflip_labels(self, labels):
        new_labels = []
        for i, label in enumerate(labels):
            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().numpy().item()
            new_labels.append(torch.as_tensor(self.flip_lut[label]))
        new_labels = torch.stack(new_labels, dim=0).to(labels.device)
        return new_labels


@ModelsManager.export("pose_transformer_semi")
class PoseTransformerSemiModel(PoseTransformerModel):
    def __init__(self, args=None, **kwargs):
        super(PoseTransformerSemiModel, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs
        self.fixmatch_threshold = dict_args.get("fixmatch_threshold")
        self.weight_unsupervised = dict_args.get("weight_unsupervised")
        self.use_negative_boxes = dict_args.get("use_negative_boxes")
        self.exclude_bg = dict_args.get("exclude_bg")

        self.fixmatch_background_threshold = dict_args.get("fixmatch_background_threshold")
        self.num_top_negative = dict_args.get("num_top_negative")
        self.sample_negative_coords = dict_args.get("sample_negative_coords")
        self.add_background_if_pos = dict_args.get("add_background_if_pos")

        self.num_classes = 17
        args.num_joints = self.num_classes

        matcher = build_coords_matcher(args)
        weight_dict = {"loss_ce": 1, "loss_coords": args.coords_loss_coef}
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

        self.pose_flipper = COCOPoseFlipper()

        losses = ["labels", "coords", "cardinality"]
        if args.masks:
            losses += ["masks"]
        criterion = CoordsSetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=args.eos_coef,
            losses=losses,
            negativ_threshold=self.fixmatch_background_threshold,
            num_top_negative=self.num_top_negative,
            add_background_if_pos=self.add_background_if_pos,
        )

        self.criterion = criterion

        self.teacher = EMA(self.model)

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.teacher:
            self.teacher.update(self.model)

    def _transform_coords(self, coords, weak_targets, strong_targets):
        # translate coords to new coordinates
        translated_coords = []
        translated_coords_abs = []
        mask_list = []
        for b in range(coords.shape[0]):

            inv_transformation = torch.linalg.inv(weak_targets["transformation"][b])
            weak_boxes_abs = point_to_abs(coords[b], size=weak_targets["size"][b])
            boxes_origins_abs = points_transformation(weak_boxes_abs, inv_transformation)

            strong_points_transformation_abs = points_transformation(
                boxes_origins_abs, strong_targets["transformation"][b]
            )

            def filter_points(x):
                if x[0] < 0 or x[1] < 0 or x[0] >= strong_targets["size"][b][1] or x[1] >= strong_targets["size"][b][0]:
                    return 1
                return 0

            mask_list.append(torch.as_tensor([filter_points(p) for p in strong_points_transformation_abs.unbind(0)]))

            strong_points_transformation = point_to_rel(
                strong_points_transformation_abs, size=strong_targets["size"][b]
            )

            translated_coords.append(strong_points_transformation)
            translated_coords_abs.append(strong_points_transformation_abs)
        out_coords = torch.stack(translated_coords, dim=0)
        out_coords_abs = torch.stack(translated_coords_abs, dim=0)
        masks = torch.stack(mask_list, dim=0)

        return out_coords, out_coords_abs, masks

    def _build_unsupervised_target(self, weak_outputs, weak_targets, strong_targets=None, include_bg=True):
        unsupervised_target = {"joints": [], "labels": [], "size": None, "scores": []}

        device = weak_outputs["pred_logits"].device
        prediction = self._get_prediction(weak_outputs, include_bg=include_bg)

        scores, labels, coords = prediction["scores"], prediction["labels"], prediction["joints"]

        if strong_targets:
            coords, out_coords_abs, masks = self._transform_coords(coords, weak_targets, strong_targets)
            new_labels = []
            label_to_flip = np.logical_xor(weak_targets["flipped"], strong_targets["flipped"])
            for b_labels, flip in zip(labels.unbind(0), label_to_flip):
                if flip:
                    flipped_labels = self.pose_flipper.hflip_labels(b_labels)

                    new_labels.append(flipped_labels)
                else:
                    new_labels.append(b_labels)

            labels = torch.stack(new_labels, dim=0)
            unsupervised_target["size"] = strong_targets["size"]
        else:
            masks = torch.zeros(coords.shape[0], coords.shape[1])
            unsupervised_target["size"] = weak_targets["size"]

        for b_scores, b_coords, b_labels, b_masks in zip(
            scores.unbind(0), coords.unbind(0), labels.unbind(0), masks.unbind(0)
        ):
            scores_list = []
            coords_list = []
            labels_list = []
            for i_score, i_coord, i_label, i_mask in zip(
                b_scores.unbind(0), b_coords.unbind(0), b_labels.unbind(0), b_masks.unbind(0)
            ):
                if i_mask == 1:
                    continue

                if i_score < self.fixmatch_threshold:
                    continue

                scores_list.append(i_score)
                coords_list.append(i_coord)
                labels_list.append(i_label)

            if len(coords_list) > 0:
                unsupervised_target["joints"].append(torch.stack(coords_list, dim=0))
                unsupervised_target["labels"].append(torch.stack(labels_list, dim=0))
                unsupervised_target["scores"].append(torch.stack(scores_list, dim=0))
            else:
                unsupervised_target["joints"].append(torch.zeros([0, 2], device=device))
                unsupervised_target["labels"].append(torch.zeros([0], dtype=torch.int64, device=device))
                unsupervised_target["scores"].append(torch.zeros([0], device=device))
        return unsupervised_target

    def training_step(self, batch, batch_idx):
        # print(flat_dict(batch).keys())
        # return
        supervised = batch["supervised"]
        samples, targets = supervised["image"], supervised["target"]
        unsupervised = batch["unsupervised"]
        weak_samples, weak_targets, strong_samples, strong_targets = (
            unsupervised["weak_image"],
            unsupervised["weak_target"],
            unsupervised["strong_image"],
            unsupervised["strong_target"],
        )
        # print(supervised["target"])
        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)

        self.log("train/supervised/count_neg", loss_dict["count_neg"])
        self.log("train/supervised/count_pos", loss_dict["count_pos"])
        self.log("train/supervised/dist", loss_dict["count_neg"] / (loss_dict["count_pos"] + 1))
        self.log("train/supervised/loss_ce", loss_dict["loss_ce"])
        self.log("train/supervised/loss_coords", loss_dict["loss_coords"])
        self.log("train/supervised/cardinality_error", loss_dict["cardinality_error"])

        weight_dict = self.criterion.weight_dict
        supervised_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        supervised_predictions = self._get_prediction(outputs, include_bg=not self.exclude_bg)
        # print("###############################")
        # print(supervised_predictions)
        # print("###############################")
        # print(self._get_prediction(outputs))
        # print(targets.keys())
        # print(targets["size"])
        # exit()

        # unsupervised transformation
        with torch.no_grad():
            weak_outputs = self.teacher(weak_samples)
            unsupervised_target = self._build_unsupervised_target(
                weak_outputs, weak_targets, strong_targets, include_bg=not self.exclude_bg
            )
        # exit()
        outputs = self.model(strong_samples)
        loss_dict = self.criterion(
            outputs,
            unsupervised_target,
            pos_only=not self.use_negative_boxes,
            sample_negative=self.sample_negative_coords,
        )

        self.log("train/unsupervised/count_neg", loss_dict["count_neg"])
        self.log("train/unsupervised/count_pos", loss_dict["count_pos"])
        self.log("train/unsupervised/dist", loss_dict["count_neg"] / (loss_dict["count_pos"] + 1))
        self.log("train/unsupervised/loss_ce", loss_dict["loss_ce"])
        self.log("train/unsupervised/loss_coords", loss_dict["loss_coords"])
        self.log("train/unsupervised/cardinality_error", loss_dict["cardinality_error"])

        weight_dict = self.criterion.weight_dict
        unsupervised_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        weak_predictions = self._build_unsupervised_target(weak_outputs, weak_targets, include_bg=not self.exclude_bg)

        self.keypoints_images = utils.detach_all(
            [
                {
                    "ids": targets["image_id"],
                    "names": "gt",
                    "images": samples.tensors,
                    "keypoints": targets["joints"],
                    "keypoints_labels": targets["labels"],
                    "sizes": targets["size"],
                },
                {
                    "ids": targets["image_id"],
                    "names": "supervised",
                    "images": samples.tensors,
                    "keypoints": supervised_predictions["joints"],
                    "keypoints_labels": supervised_predictions["labels"],
                    "keypoints_scores": supervised_predictions["scores"],
                    "sizes": targets["size"],
                },
                {
                    "ids": weak_targets["image_id"],
                    "names": "unsupervised_weak",
                    "images": weak_samples.tensors,
                    "keypoints": weak_predictions["joints"],
                    "keypoints_labels": weak_predictions["labels"],
                    "keypoints_scores": weak_predictions["scores"],
                    "sizes": weak_targets["size"],
                },
                {
                    "ids": strong_targets["image_id"],
                    "names": "unsupervised_strong",
                    "images": strong_samples.tensors,
                    "keypoints": unsupervised_target["joints"],
                    "keypoints_labels": unsupervised_target["labels"],
                    "keypoints_scores": unsupervised_target["scores"],
                    "sizes": strong_targets["size"],
                },
            ]
        )
        self.log("train/supervised_loss", supervised_losses)
        self.log("train/unsupervised_loss", self.weight_unsupervised * unsupervised_losses)
        self.log("train/loss", supervised_losses + self.weight_unsupervised * unsupervised_losses)

        return {"loss": supervised_losses + self.weight_unsupervised * unsupervised_losses}

    @classmethod
    def add_args(cls, parent_parser):
        print("PoseTransformerSemi::add_args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser = super(PoseTransformerSemiModel, cls).add_args(parser)

        parser.add_argument("--threshold", type=float, default=0.9)

        parser.add_argument("--coords_loss_coef", default=5, type=float)
        parser.add_argument("--set_cost_coord", default=5, type=float)
        parser.add_argument("--prtr_resume_path")
        parser.add_argument("--backbone_arch", default="resnet50")
        parser.add_argument("--keypoint_threshold", default=0.9, type=float)

        parser.add_argument("--fixmatch_threshold", type=float, default=0.9)

        parser.add_argument("--fixmatch_background_threshold", type=float)
        parser.add_argument("--num_top_negative", type=int, default=5)
        parser.add_argument("--exclude_bg", action="store_true")

        parser.add_argument("--sample_negative_coords", action="store_true")
        parser.add_argument("--add_background_if_pos", action="store_true")

        parser.add_argument("--use_negative_boxes", action="store_true")
        parser.add_argument(
            "--weight_unsupervised", default=1, type=float, help="L1 box coefficient in the matching cost"
        )

        return parser

    @classmethod
    def tunning_scopes(cls):
        from ray import tune

        parameter_scope = {}

        if hasattr(super(PoseTransformerSemiModel, cls), "tunning_scopes"):
            parameter_scope.update(super(PoseTransformerSemiModel, cls).tunning_scopes())
        parameter_scope.update(
            {
                # "fixmatch_threshold": tune.choice([0.8, 0.9, 0.95, 0.99]),
                "weight_unsupervised": tune.choice([0.5, 1.0, 2.0, 5.0, 10.0]),
            }
        )
        return parameter_scope