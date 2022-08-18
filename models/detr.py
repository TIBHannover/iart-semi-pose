# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

import utils.misc as utils
from utils import box_ops
from utils.misc import NestedTensor, detach_all, nested_tensor_from_tensor_list, flat_dict, unflat_dict

from .backbone import build_backbone, BackboneConfig
from .matcher import build_boxes_matcher
from .losses import BoxesSetCriterion
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from .transformer import build_transformer

from datasets.coco_eval import CocoEvaluator
from datasets.coco_eval import COCO

from pytorch_lightning.utilities.distributed import rank_zero_only


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
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
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
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


import logging
import argparse

from models.models import ModelsManager
from pytorch_lightning.core.lightning import LightningModule

from utils.box_ops import (
    boxes_transformation,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    boxes_to_abs,
    boxes_to_rel,
    boxes_fit_size,
)


@ModelsManager.export("detr")
class DETRModel(LightningModule):
    def __init__(self, args=None, **kwargs):
        super(DETRModel, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.masks = dict_args.get("masks")

        self.aux_loss = dict_args.get("aux_loss")

        self.set_cost_class = dict_args.get("set_cost_class")
        self.set_cost_bbox = dict_args.get("set_cost_bbox")
        self.set_cost_giou = dict_args.get("set_cost_giou")
        self.mask_loss_coef = dict_args.get("mask_loss_coef")
        self.dice_loss_coef = dict_args.get("dice_loss_coef")
        self.bbox_loss_coef = dict_args.get("bbox_loss_coef")
        self.giou_loss_coef = dict_args.get("giou_loss_coef")
        self.eos_coef = dict_args.get("eos_coef")
        self.enc_layers = dict_args.get("enc_layers")
        self.dec_layers = dict_args.get("dec_layers")
        self.dim_feedforward = dict_args.get("dim_feedforward")
        self.hidden_dim = dict_args.get("hidden_dim")
        self.dropout = dict_args.get("dropout")
        self.nheads = dict_args.get("nheads")
        self.num_queries = dict_args.get("num_queries")
        self.pre_norm = dict_args.get("pre_norm")
        self.backbone = dict_args.get("backbone")
        self.dilation = dict_args.get("dilation")
        self.position_embedding = dict_args.get("position_embedding")
        self.frozen_weights = dict_args.get("frozen_weights")
        self.lr = dict_args.get("lr")
        self.lr_backbone = dict_args.get("lr_backbone")
        self.batch_size = dict_args.get("batch_size")
        self.weight_decay = dict_args.get("weight_decay")
        self.lr_drop = dict_args.get("lr_drop")
        self.clip_max_norm = dict_args.get("clip_max_norm")

        self.label_whitelist = dict_args.get("label_whitelist")

        self.val_gt_annotation_path = dict_args.get("val_gt_annotation_path")
        self.val_gt = None
        if self.val_gt_annotation_path:
            self.val_gt = COCO(self.val_gt_annotation_path)

        self.test_gt_annotation_path = dict_args.get("test_gt_annotation_path")
        self.test_gt = None
        if self.test_gt_annotation_path:
            self.test_gt = COCO(self.test_gt_annotation_path)

        self.detr_resume_path = dict_args.get("detr_resume_path")
        self.reinit_heads = dict_args.get("reinit_heads")
        self.reinit_classifier = dict_args.get("reinit_classifier")
        self.reinit_last_heads = dict_args.get("reinit_last_heads")

        # the `num_classes` naming here is somewhat misleading.
        # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        # is the maximum id for a class in your dataset. For example,
        # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
        # As another example, for a dataset that has a single class with id 1,
        # you should pass `num_classes` to be 2 (max_obj_id + 1).
        # For more details on this, check the following discussion
        # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
        # num_classes = 20 if args.dataset_file != "coco" else 91
        # if args.dataset_file == "coco_panoptic":
        #     # for panoptic, we just add a num_classes that is large enough to hold
        #     # max_obj_id + 1, but the exact value doesn't really matter
        #     num_classes = 250
        # device = torch.device(args.device)

        num_classes = 91

        backbone = build_backbone(
            BackboneConfig(
                lr_backbone=self.lr_backbone,
                masks=self.masks,
                backbone=self.backbone,
                dilation=self.dilation,
                hidden_dim=self.hidden_dim,
                position_embedding=self.position_embedding,
            )
        )

        transformer = build_transformer(args)

        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=self.num_queries,
            aux_loss=args.aux_loss,
        )
        if args.masks:
            model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
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
        criterion = BoxesSetCriterion(
            num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses
        )
        # criterion.to(device)
        postprocessors = {"bbox": PostProcess()}
        if args.masks:
            postprocessors["segm"] = PostProcessSegm()
            if args.dataset_file == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

        self.criterion = criterion
        self.model = model
        self.postprocessors = postprocessors

        # TODO move away
        self.val_coco_evaluator = None
        if self.val_gt:
            iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors.keys())
            self.val_coco_evaluator = CocoEvaluator(self.val_gt, iou_types, label_whitelist=self.label_whitelist)
        # TODO move away
        self.test_coco_evaluator = None
        if self.test_gt:
            iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors.keys())
            self.test_coco_evaluator = CocoEvaluator(self.test_gt, iou_types, label_whitelist=self.label_whitelist)

        if args.detr_resume_path is not None:
            self.load(self.detr_resume_path)

    def forward(self, x):
        return x

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
        samples, targets = batch["image"], batch["target"]

        outputs = self.model(samples)
        loss_dict = detach_all(self.criterion(outputs, targets))
        weight_dict = self.criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        # metric_logger.update(
        #     loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        # )
        # metric_logger.update(class_error=loss_dict_reduced["class_error"])
        # print(targets)
        # exit()
        orig_target_sizes = torch.stack(targets["orig_size"], dim=0)
        # print(orig_target_sizes)
        results = self.postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in self.postprocessors.keys():
            target_sizes = torch.stack(targets["size"], dim=0)
            results = self.postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

        # move everything away from the gpu
        results = detach_all(results)
        targets = detach_all(targets)
        res = {target.item(): output for target, output in zip(targets["image_id"], results)}
        if self.val_coco_evaluator is not None:
            self.val_coco_evaluator.update(res)

        return {"loss": sum(loss_dict_reduced_scaled.values())}

    def _log_coco(self, results):
        for k, v in results.items():
            if k == "bbox":
                assert len(v) == 12

                self.log(f"val/map", v[0], prog_bar=True)

                self.log("val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", v[0])
                self.log("val/bbox/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]", v[1])
                self.log("val/bbox/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]", v[2])
                self.log("val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", v[3])
                self.log("val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", v[4])
                self.log("val/bbox/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", v[5])
                self.log("val/bbox/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]", v[6])
                self.log("val/bbox/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]", v[7])
                self.log("val/bbox/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", v[8])
                self.log("val/bbox/Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", v[9])
                self.log("val/bbox/Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", v[10])
                self.log("val/bbox/Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", v[11])

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

        if self.val_coco_evaluator is not None:
            self.val_coco_evaluator.synchronize_between_processes()
        if self.val_coco_evaluator is not None:
            self.val_coco_evaluator.accumulate()
            result_dict = self.val_coco_evaluator.summarize()
            self._log_coco(result_dict)
        if self.val_gt:
            iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors.keys())
            self.val_coco_evaluator = CocoEvaluator(self.val_gt, iou_types, label_whitelist=self.label_whitelist)

    def test_step(self, batch, batch_idx):
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
        # print(targets)
        # exit()
        orig_target_sizes = torch.stack(targets["orig_size"], dim=0)
        # print(orig_target_sizes)
        results = self.postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in self.postprocessors.keys():
            target_sizes = torch.stack(targets["size"], dim=0)
            results = self.postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

        # move everything away from the gpu
        results = detach_all(results)
        targets = detach_all(targets)
        res = {target.item(): output for target, output in zip(targets["image_id"], results)}
        if self.test_coco_evaluator is not None:
            self.test_coco_evaluator.update(res)

        return {"loss": sum(loss_dict_reduced_scaled.values())}

    def test_epoch_end(self, outputs):
        logging.info("DETR::test_epoch_end")
        loss = 0.0
        count = 0
        for entry in outputs:
            for k, v in entry.items():
                if k == "loss":
                    loss += v
                    count += 1
        self.log("val/loss", loss / count)

        if self.test_coco_evaluator is not None:
            self.test_coco_evaluator.synchronize_between_processes()
        if self.test_coco_evaluator is not None:
            self.test_coco_evaluator.accumulate()
            result_dict = self.test_coco_evaluator.summarize()
            self._log_coco(result_dict)

        iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors.keys())
        self.coco_evaluator = CocoEvaluator(self.val_gt, iou_types, label_whitelist=self.label_whitelist)

    def _post_process_predictions(self, outputs, targets, threshold=0.9):

        predictions = {"boxes": [], "labels": [], "size": targets["size"], "scores": []}
        # print(supervised["target"])
        # exit()

        batch_size = outputs["pred_logits"].shape[0]

        # print(weak_outputs)
        # print(weak_outputs['pred_logits'].shape)
        label_softmax = torch.softmax(outputs["pred_logits"], dim=-1)
        # print(label_softmax.shape)
        # exit()
        top_prediction = label_softmax > threshold
        boxes_pos = top_prediction[..., :-1].nonzero()
        # print(boxes_pos)
        # exit()
        # transform_cordinates
        for b in range(batch_size):
            boxes = []
            labels = []
            scores = []
            inv_transformation = torch.linalg.inv(targets["transformation"][b])
            weak_boxes_abs = boxes_to_abs(box_cxcywh_to_xyxy(outputs["pred_boxes"][b]), size=targets["size"][b])
            boxes_origins_abs = boxes_transformation(weak_boxes_abs, inv_transformation)

            boxes_sample = boxes_pos[boxes_pos[:, 0] == b]

            for box in boxes_sample.unbind(0):
                box_index = box[1]
                box_cls = box[2]
                box_cxcywh = boxes_origins_abs[box_index]
                box_score = label_softmax[b, box_index, box_cls]
                if self.label_whitelist is None:
                    labels.append(box_cls)
                    boxes.append(box_cxcywh)
                    scores.append(box_score)
                else:
                    box_label = box_cls.detach().cpu().numpy().item()
                    if box_label in self.label_whitelist:

                        # print(box_label)
                        labels.append(box_cls)
                        boxes.append(box_cxcywh)
                        scores.append(box_score)
                    # exit()
            #     print(f"{box} {box_index} {box_cls} {box_cxcywh}")
            # print(boxes)
            # print(labels)
            if len(boxes) > 0:
                predictions["boxes"].append(torch.stack(boxes, dim=0))
                predictions["labels"].append(torch.stack(labels, dim=0))
                predictions["scores"].append(torch.stack(scores, dim=0))
            else:
                predictions["boxes"].append(
                    torch.zeros(
                        [0, 4],
                        device=label_softmax.device,
                    )
                )
                predictions["labels"].append(torch.zeros([0], dtype=torch.int64, device=label_softmax.device))
                predictions["scores"].append(torch.zeros([0], device=label_softmax.device))
        return predictions

    def infer_step(self, batch, threshold=0.9):

        outputs = self.model(batch["image"])
        predictions = self._post_process_predictions(outputs, batch["target"], threshold=threshold)
        predictions = detach_all(predictions)

        return {**batch["target"], **predictions}

    def configure_optimizers(self):

        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_drop)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]

    def load(self, checkpoint_path):
        logging.info("Loading checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if self.reinit_classifier:
            checkpoint = unflat_dict(checkpoint["model"])
            del checkpoint["class_embed"]["weight"]
            del checkpoint["class_embed"]["bias"]
            checkpoint = {"model": flat_dict(checkpoint)}
        elif self.reinit_heads:
            checkpoint = unflat_dict(checkpoint["model"])
            del checkpoint["class_embed"]
            del checkpoint["bbox_embed"]
            checkpoint = {"model": flat_dict(checkpoint)}
        elif self.reinit_last_heads:
            checkpoint = unflat_dict(checkpoint["model"])
            del checkpoint["bbox_embed"]["layers"]["2"]["weight"]
            del checkpoint["bbox_embed"]["layers"]["2"]["bias"]
            del checkpoint["class_embed"]["weight"]
            del checkpoint["class_embed"]["bias"]
            checkpoint = {"model": flat_dict(checkpoint)}

        self.model.load_state_dict(checkpoint["model"], strict=False)

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("DETR::add_args")
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")

        parser.add_argument(
            "--no_aux_loss",
            dest="aux_loss",
            action="store_false",
            help="Disables auxiliary decoding losses (loss at each layer)",
        )

        # segmentation
        parser.add_argument("--masks", action="store_true", help="Train segmentation head if the flag is provided")

        # matcher
        parser.add_argument("--set_cost_class", default=1, type=float, help="Class coefficient in the matching cost")
        parser.add_argument("--set_cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost")
        parser.add_argument("--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")

        # loss coefficients
        parser.add_argument("--mask_loss_coef", default=1, type=float)
        parser.add_argument("--dice_loss_coef", default=1, type=float)
        parser.add_argument("--bbox_loss_coef", default=5, type=float)
        parser.add_argument("--giou_loss_coef", default=2, type=float)
        parser.add_argument(
            "--eos_coef", default=0.1, type=float, help="Relative classification weight of the no-object class"
        )

        # transformer
        parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
        parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
        parser.add_argument(
            "--dim_feedforward",
            default=2048,
            type=int,
            help="Intermediate size of the feedforward layers in the transformer blocks",
        )
        parser.add_argument(
            "--hidden_dim", default=256, type=int, help="Size of the embeddings (dimension of the transformer)"
        )
        parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
        parser.add_argument(
            "--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions"
        )
        parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
        parser.add_argument("--pre_norm", action="store_true")

        # backbone
        parser.add_argument(
            "--backbone", default="resnet50", type=str, help="Name of the convolutional backbone to use"
        )
        parser.add_argument(
            "--dilation",
            action="store_true",
            help="If true, we replace stride with dilation in the last convolutional block (DC5)",
        )
        parser.add_argument(
            "--position_embedding",
            default="sine",
            type=str,
            choices=("sine", "learned"),
            help="Type of positional embedding to use on top of the image features",
        )

        # model
        parser.add_argument(
            "--frozen_weights",
            type=str,
            default=None,
            help="Path to the pretrained model. If set, only the mask head will be trained",
        )

        # opt
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--lr_backbone", default=1e-5, type=float)
        parser.add_argument("--batch_size", default=2, type=int)
        parser.add_argument("--weight_decay", default=1e-4, type=float)
        parser.add_argument("--lr_drop", default=150000, type=int)
        parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")

        # loading
        parser.add_argument("--detr_resume_path", help="resume from checkpoint")
        parser.add_argument("--reinit_heads", action="store_true", help="resum from checkpoint")
        parser.add_argument("--reinit_last_heads", action="store_true", help="resum from checkpoint")
        parser.add_argument("--reinit_classifier", action="store_true", help="resum from checkpoint")

        # val
        parser.add_argument("--val_gt_annotation_path", type=str)
        parser.add_argument("--label_whitelist", type=int, nargs="+")

        # test
        parser.add_argument("--test_gt_annotation_path", type=str)
        return parser

    @classmethod
    def tunning_scopes(cls):
        from ray import tune

        parameter_scope = {}

        if hasattr(super(DETRModel, cls), "tunning_scopes"):
            parameter_scope.update(super(DETRModel, cls).tunning_scopes())
        parameter_scope.update(
            {
                # "lr": tune.choice([1e-4, 5e-5, 1e-5, 5e-6, 1e-6]),
                # "lr_backbone": tune.choice([1e-5, 1e-6, 1e-7, 1e-8]),
                # "weight_decay": tune.choice([1e-4, 1e-5]),
            }
        )
        return parameter_scope