# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_4points(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0, x1, y1, x0, y1, x1, y0]
    return torch.stack(b, dim=-1)


def box_4points_to_xyxy(x):
    x0, y0, x1, y1, x2, y2, x3, y3 = x.unbind(-1)
    xs = torch.stack([x0, x1, x2, x3], dim=-1)
    ys = torch.stack([y0, y1, y2, y3], dim=-1)

    b = [
        torch.min(xs, dim=-1).values,
        torch.min(ys, dim=-1).values,
        torch.max(xs, dim=-1).values,
        torch.max(ys, dim=-1).values,
    ]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# modified from torchvision to also return the union
def points_in_box(boxes, points):
    if torch.sum(points < boxes[0:2]):
        return False

    if torch.sum(points > boxes[2:4]):
        return False

    return True


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def point_to_abs(points, size):

    original_shape = points.shape
    points = points.reshape(-1, original_shape[-1])
    points, meta = points[:, :2], points[:, 2:]
    points = points * torch.as_tensor([size[1], size[0]], device=points.device)

    transformed_points = torch.cat([points, meta], dim=-1)
    return transformed_points.reshape(original_shape)


def point_to_rel(points, size):

    original_shape = points.shape
    points = points.reshape(-1, original_shape[-1])
    points, meta = points[:, :2], points[:, 2:]
    points = points / torch.as_tensor([size[1], size[0]], device=points.device)

    transformed_points = torch.cat([points, meta], dim=-1)
    return transformed_points.reshape(original_shape)


def points_transformation(points, transformation):
    original_shape = points.shape

    # prepare points [N,3]
    points = points.reshape(-1, original_shape[-1])
    points, meta = points[:, :2], points[:, 2:]
    points = torch.cat([points, torch.as_tensor([[1.0]] * points.shape[0], device=points.device)], dim=1)
    points = points.unsqueeze(2)

    # prepare transformation [N,3,3]
    if len(transformation.shape) == 2:
        transformation = torch.unsqueeze(transformation, dim=0).expand(points.shape[0], 3, 3)

    transformed_points = transformation @ points
    transformed_points = torch.cat([torch.squeeze(transformed_points, 2)[:, :2], meta], dim=-1)
    return transformed_points.reshape(original_shape)


def boxes_to_abs(boxes, size):

    original_shape = boxes.shape
    boxes = boxes.reshape(-1, original_shape[-1])
    boxes, meta = boxes[:, :4], boxes[:, 4:]
    boxes = boxes * torch.as_tensor([size[1], size[0], size[1], size[0]], device=boxes.device)

    transformed_boxes = torch.cat([boxes, meta], dim=-1)
    return transformed_boxes.reshape(original_shape)


def boxes_to_rel(boxes, size):

    original_shape = boxes.shape
    boxes = boxes.reshape(-1, original_shape[-1])
    boxes, meta = boxes[:, :4], boxes[:, 4:]
    boxes = boxes / torch.as_tensor([size[1], size[0], size[1], size[0]], device=boxes.device)

    transformed_boxes = torch.cat([boxes, meta], dim=-1)
    return transformed_boxes.reshape(original_shape)


def boxes_transformation(boxes, transformation):
    original_shape = boxes.shape

    # prepare points [N,3]
    points = boxes.reshape(-1, original_shape[-1])
    # should be possible with a single reshape
    points_xyxy, meta = points[:, :4], points[:, 4:]
    # we need to compute all 4 points

    points = box_xyxy_to_4points(points_xyxy)
    points = points.reshape(-1, 2)
    transformed_points = points_transformation(points, transformation)
    transformed_points = transformed_points.reshape(-1, 8)

    transformed_points = box_4points_to_xyxy(transformed_points)

    transformed_points = torch.cat([transformed_points, meta], dim=-1)
    return transformed_points.reshape(original_shape)


def boxes_fit_size(boxes, size):
    h, w = size[0], size[1]

    original_shape = boxes.shape

    max_size = torch.as_tensor([w, h], dtype=torch.float32, device=size.device)
    boxes = torch.min(boxes.reshape(-1, 2, 2), max_size)
    boxes = boxes.clamp(min=0)

    return boxes.reshape(original_shape)


def boxes_scale(boxes, scale, size=None):

    box_cxcywh = box_xyxy_to_cxcywh(boxes)
    scaled_box_wh = box_cxcywh[2:] * scale
    scaled_box = box_cxcywh_to_xyxy(torch.cat([box_cxcywh[:2], scaled_box_wh], dim=0))
    if size is not None:
        scaled_box = boxes_fit_size(scaled_box, size)

    return scaled_box


def boxes_aspect_ratio(boxes, aspect_ratio, size=None):
    box_cxcywh = box_xyxy_to_cxcywh(boxes)
    w, h = box_cxcywh[2], box_cxcywh[3]
    n_w, n_h = w, h
    if w > aspect_ratio * h:
        n_h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        n_w = h * aspect_ratio
    scaled_box = box_cxcywh_to_xyxy(torch.stack([box_cxcywh[0], box_cxcywh[1], n_w, n_h], dim=0))
    if size is not None:
        scaled_box = boxes_fit_size(scaled_box, size)
    return scaled_box
