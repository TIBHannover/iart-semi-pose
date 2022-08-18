# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou



class HungarianBoxesMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat(targets["labels"])
        tgt_bbox = torch.cat(targets["boxes"])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets["boxes"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_boxes_matcher(args):
    return HungarianBoxesMatcher(
        cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou
    )


# class HungarianCoordsMatcher(nn.Module):
#     """This class computes an assignment between the targets and the predictions of the network
#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """

#     def __init__(self, num_joints, cost_class: float = 1, cost_coord: float = 1):
#         """Creates the matcher
#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_coord: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
#         """
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_coord = cost_coord
#         self.num_joints = num_joints
#         assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         ## target: [bs, 17, 2]
#         bs, num_queries = outputs["pred_logits"].shape[:2]

#         # We flatten to compute the cost matrices in a batch
#         out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
#         out_kpt = outputs["pred_coords"]  # [batch_size, num_queries, 2]

#         tgt_ids = torch.cat(targets["labels"])
#         print('+++++++++++++')
#         print(tgt_ids)
#         tgt_joints = torch.cat(targets["joints"])
#         print(tgt_joints.shape)
#         # Compute the classification cost. Contrary to the loss, we don't use the NLL,
#         # but approximate it in 1 - proba[target class].
#         # The 1 is a constant that doesn't change the matching, it can be ommitted.
#         cost_class = -out_prob[...,tgt_ids]
#         print(cost_class.shape)
#         print(out_kpt.shape)
#         # Compute the L1 cost between keypoints
#         cost_kpt = torch.cdist(out_kpt, tgt_joints, p=1)  # [B, N, 17]

#         print(cost_class.shape)
#         print(cost_kpt.shape)
#         # Final cost matrix
#         print('############')
#         print(cost_kpt.shape)
#         print(self.cost_coord)
#         C = self.cost_coord * cost_kpt
#         print('############')
#         print(cost_class.shape)
#         print(self.cost_class)
#         B= self.cost_class * cost_class
#         C = C + B
#         C = C.transpose(1, 2).cpu()  # [B, 17, N]

#         indices = [linear_sum_assignment(c) for c in C]
#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]




class HungarianCoordsMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_coord: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_coord: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_coord = cost_coord
        assert cost_class != 0 or cost_bbox != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_coords"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat(targets["labels"])
        tgt_bbox = torch.cat(targets["joints"])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class #+ self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets["joints"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



def build_coords_matcher(args):
    return HungarianCoordsMatcher( cost_class=args.set_cost_class, cost_coord=args.set_cost_coord)
    # return HungarianCoordsMatcher(args.num_joints, cost_class=args.set_cost_class, cost_coord=args.set_cost_coord)