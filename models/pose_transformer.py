import logging
import argparse
import copy

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


from utils.misc import NestedTensor, detach_all, nested_tensor_from_tensor_list

from utils.box_ops import points_transformation, point_to_abs


from .pose_backbone import build_backbone
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from .transformer import build_transformer
from .matcher import build_coords_matcher
from .losses import CoordsSetCriterion

from datasets.coco_eval import CocoKeypointsEvaluator


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_coords": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.coord_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_coords": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_coords": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, threshold):

        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, outputs, targets, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_coords = outputs["pred_logits"], outputs["pred_coords"]
        translated_coords = []
        for b in range(len(targets["transformation"])):
            # print(b)
            # print(out_coords.shape)
            inv_transformation = torch.linalg.inv(targets["transformation"][b])
            coords_abs = point_to_abs(out_coords[b], targets["size"][b])
            coords_origin_abs = points_transformation(coords_abs, inv_transformation)
            translated_coords.append(coords_origin_abs)
            # print(coords_origin_abs)
        out_coords = torch.stack(translated_coords, dim=0)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        num_joints = out_logits.shape[-1] - 1

        prob = F.softmax(out_logits, -1)

        prob_cpu = F.softmax(out_logits[..., :-1], dim=-1).detach().cpu()

        _, labels = prob[..., :-1].max(-1)

        scores_list = []
        coords_list = []
        labels_list = []
        for b, C in enumerate(prob_cpu):

            _, query_ind = linear_sum_assignment(-C.transpose(0, 1))  # Cost Matrix: [17, N]
            score = prob_cpu[b, query_ind, list(np.arange(num_joints))].numpy()

            coord = out_coords[b, query_ind].detach().cpu().numpy()
            scores_list.append(torch.as_tensor(score))
            coords_list.append(torch.as_tensor(coord))
            labels_list.append(labels[b, query_ind])
        scores = torch.stack(scores_list)
        coords = torch.stack(coords_list)
        labels = torch.stack(labels_list)

        results = [
            {"scores": s, "labels": l, "keypoints": b, "selected": s > self.threshold}
            for s, l, b in zip(scores, labels, coords)
        ]

        return results


@ModelsManager.export("pose_transformer")
class PoseTransformerModel(DETRModel):
    def __init__(self, args=None, **kwargs):
        super(PoseTransformerModel, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.coords_loss_coef = dict_args.get("coords_loss_coef")
        self.set_cost_coord = dict_args.get("set_cost_coord")
        self.prtr_resume_path = dict_args.get("prtr_resume_path")
        self.backbone_arch = dict_args.get("backbone_arch")
        self.keypoint_threshold = dict_args.get("keypoint_threshold")

        self.num_classes = 17
        args.num_joints = self.num_classes

        backbone = build_backbone(self.backbone_arch)

        transformer = build_transformer(args)

        model = DETR(
            backbone,
            transformer,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            aux_loss=args.aux_loss,
        )
        if args.masks:
            model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
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

        losses = ["labels", "coords", "cardinality"]
        if args.masks:
            losses += ["masks"]
        criterion = CoordsSetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses
        )
        # criterion.to(device)
        postprocessors = {"keypoints": PostProcess(threshold=self.keypoint_threshold)}
        if args.masks:
            postprocessors["segm"] = PostProcessSegm()
            if args.dataset_file == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

        self.criterion = criterion
        self.model = model
        self.postprocessors = postprocessors

        # TODO move away
        iou_types = tuple(k for k in ("segm", "bbox", "keypoints") if k in self.postprocessors.keys())
        self.coco_evaluator = CocoKeypointsEvaluator(self.val_gt, iou_types)

        if args.prtr_resume_path is not None:
            self.load(self.prtr_resume_path)

    def load(self, checkpoint_path):
        logging.info("Loading checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        unflatten_checkpoint = utils.unflat_dict(checkpoint)
        self.model.transformer.load_state_dict(utils.flat_dict(unflatten_checkpoint["transformer"]))

        self.model.input_proj.load_state_dict(utils.flat_dict(unflatten_checkpoint["input_proj"]))

        self.model.query_embed.load_state_dict(utils.flat_dict(unflatten_checkpoint["query_embed"]))

        if not self.reinit_heads:
            if self.reinit_classifier:
                self.model.coord_embed.load_state_dict(utils.flat_dict(unflatten_checkpoint["kpt_embed"]))
            elif self.reinit_last_heads:
                # we skip class in this case
                kpts_embed = unflatten_checkpoint["kpt_embed"]
                del kpts_embed["layers"]["2"]["weight"]
                del kpts_embed["layers"]["2"]["bias"]

                self.model.coord_embed.load_state_dict(utils.flat_dict(kpts_embed), strict=False)
            else:
                self.model.class_embed.load_state_dict(utils.flat_dict(unflatten_checkpoint["class_embed"]))
                self.model.coord_embed.load_state_dict(utils.flat_dict(unflatten_checkpoint["kpt_embed"]))

        self.model.backbone[0].body.load_state_dict(utils.flat_dict(unflatten_checkpoint["backbone"]["body"]))

    def _get_prediction(self, outputs, include_bg=False):
        pred_logits = outputs["pred_logits"].float().detach().cpu()
        pred_coords = outputs["pred_coords"].float().detach().cpu()

        device = outputs["pred_logits"].device

        num_joints = pred_logits.shape[-1] - 1

        if include_bg:
            prob = F.softmax(pred_logits, dim=-1)[..., :-1]
        else:
            prob = F.softmax(pred_logits[..., :-1], dim=-1)
        score_holder = []
        coord_holder = []
        labels_holder = []
        orig_coord = []
        for b, C in enumerate(prob):
            # print(C)
            _, query_ind = linear_sum_assignment(-C.transpose(0, 1))  # Cost Matrix: [17, N]
            score = prob[b, query_ind, list(np.arange(num_joints))][..., None].numpy()
            pred_raw = pred_coords[b, query_ind].numpy()
            # scale to the whole patch
            # pred_raw *= np.array(config.MODEL.IMAGE_SIZE)
            # # transform back w.r.t. the entire img
            # pred = transform_preds(pred_raw, center[b], scale[b], config.MODEL.IMAGE_SIZE)
            orig_coord.append(torch.as_tensor(pred_raw, device=device))
            score_holder.append(torch.as_tensor(score, device=device))
            labels_holder.append(torch.as_tensor(np.arange(num_joints), device=device))
            # coord_holder.append(pred)

        matched_score = torch.stack(score_holder)
        matched_labels = torch.stack(labels_holder)
        # matched_coord = np.stack(coord_holder)

        return {"joints": torch.stack(orig_coord), "labels": matched_labels, "scores": matched_score}

    def training_step(self, batch, batch_idx):
        samples, targets = batch["image"], batch["target"]

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        self.log("train/supervised_loss", losses)
        self.log("train/loss", losses)

        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        logging.info(f"VALIDATION_STEP START")
        samples, targets = batch["image"], batch["target"]

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        # metric_logger.update(
        #     loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        # )
        # metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack(targets["orig_size"], dim=0)
        results = self.postprocessors["keypoints"](outputs, targets, orig_target_sizes)

        if "segm" in self.postprocessors.keys():
            target_sizes = torch.stack(targets["size"], dim=0)
            results = self.postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)
        # print(targets["image_id"])
        res = [{target.item(): output} for target, output in zip(targets["image_id"], results)]
        # print(res)
        # print(len(res))
        # exit()
        if self.coco_evaluator is not None:
            self.coco_evaluator.update(res)

        logging.info(f"VALIDATION_STEP END")
        return {"loss": sum(loss_dict_reduced_scaled.values())}

    def _log_coco(self, results):
        for k, v in results.items():
            if k == "keypoints":
                assert len(v) == 10

                self.log(f"val/map", v[0], prog_bar=True)
                self.log("val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]", v[0])
                self.log("val/kpt/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]", v[1])
                self.log("val/kpt/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]", v[2])
                self.log("val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]", v[3])
                self.log("val/kpt/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]", v[4])
                self.log("val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]", v[5])
                self.log("val/kpt/Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]", v[6])
                self.log("val/kpt/Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]", v[7])
                self.log("val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]", v[8])
                self.log("val/kpt/Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]", v[9])

    def validation_epoch_end(self, outputs):
        logging.info("DETR::validation_epoch_end")
        loss = 0.0
        count = 0
        for entry in outputs:
            for k, v in entry.items():
                if k == "loss":
                    loss += v
                    count += 1
        self.log("val/loss", loss / count)

        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
        if self.coco_evaluator is not None:
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            result_dict = self.coco_evaluator.summarize()
            self._log_coco(result_dict)

        iou_types = tuple(k for k in ("segm", "bbox", "keypoints") if k in self.postprocessors.keys())
        self.coco_evaluator = CocoKeypointsEvaluator(self.val_gt, iou_types)
        # exit()

    def _post_process_predictions(self, outputs, targets, threshold=0.9):

        postprocessor = PostProcess(threshold=threshold)
        target_size = torch.stack(targets["size"], dim=0)
        predictions = postprocessor(outputs, targets, target_size)
        fields = ["scores", "labels", "keypoints", "selected"]

        output_results = {}
        for field in fields:
            output_results[field] = [x[field] for x in predictions]
        # print(output_results)
        return output_results

    def infer_step(self, batch, threshold=0.9):
        # print(batch["image"].tensors.shape)
        outputs = self.model(batch["image"])
        predictions = self._post_process_predictions(outputs, batch["target"], threshold)
        predictions = detach_all(predictions)
        return {**batch["target"], **predictions}

    @classmethod
    def add_args(cls, parent_parser):
        print("PoseTransformer::add_args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser = super(PoseTransformerModel, cls).add_args(parser)

        parser.add_argument("--threshold", type=float, default=0.9)

        parser.add_argument("--coords_loss_coef", default=5, type=float)
        parser.add_argument("--set_cost_coord", default=5, type=float)
        parser.add_argument("--prtr_resume_path")
        parser.add_argument("--backbone_arch", default="resnet50")
        parser.add_argument("--keypoint_threshold", default=0.9, type=float)

        return parser

    @classmethod
    def tunning_scopes(cls):
        from ray import tune

        parameter_scope = {}

        if hasattr(super(PoseTransformerModel, cls), "tunning_scopes"):
            parameter_scope.update(super(PoseTransformerModel, cls).tunning_scopes())
        parameter_scope.update({})
        return parameter_scope