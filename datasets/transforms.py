# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import math
import copy
from numpy.core.fromnumeric import transpose
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.functional_pil as F_pil
import torchvision.transforms.functional_tensor as F_t


from typing import List, Tuple, Any, Optional

from utils.box_ops import box_xyxy_to_cxcywh
from utils.misc import interpolate

pil_modes_mapping = {
    "NEAREST": 0,
    "BILINEAR": 2,
    "BICUBIC": 3,
    "BOX": 4,
    "HAMMING": 5,
    "LANCZOS": 1,
}


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = [x for x in ["labels", "area", "iscrowd"] if x in target]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "keypoints" in target:
        keypoints = target["keypoints"]

        moved_keypoints = keypoints - torch.as_tensor([j, i, 0])
        # TODO make this better
        for p in range(moved_keypoints.shape[0]):
            for k in range(moved_keypoints.shape[1]):
                if (
                    moved_keypoints[p, k, 0] < 0
                    or moved_keypoints[p, k, 1] < 0
                    or moved_keypoints[p, k, 0] >= w
                    or moved_keypoints[p, k, 1] >= h
                ):
                    moved_keypoints[p, k] = torch.tensor([0, 0, 0])

        target["keypoints"] = moved_keypoints
        fields.append("keypoints")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            try:
                target[field] = target[field][keep]
            except:
                print(keep)
                print(target[field])
                print(field)
                exit()

    new_transformation = _get_inverse_affine_matrix(
        center=[h * 0.5, w * 0.5],
        angle=0,
        translate=[-j, -i],
        scale=[1.0, 1.0],
        shear=[0, 0],
    )

    new_transformation = torch.as_tensor([new_transformation + [0, 0, 1]])
    new_transformation = torch.linalg.inv(new_transformation.reshape(3, 3))

    if "transformation" in target:
        target_tranformation = target["transformation"]
    else:
        target_tranformation = torch.eye(3)

    target["transformation"] = new_transformation @ target_tranformation

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size
    new_transformation = _get_inverse_affine_matrix(
        center=[w / 2, 0],
        angle=0.0,
        translate=[0, 0],
        scale=[-1.0, 1.0],
        shear=[0.0, 0.0],
    )
    new_transformation = torch.as_tensor([new_transformation + [0, 0, 1]])
    new_transformation = torch.linalg.inv(new_transformation.reshape(3, 3))

    if "transformation" in target:
        target_tranformation = target["transformation"]
    else:
        target_tranformation = torch.eye(3)

    target["transformation"] = new_transformation @ target_tranformation

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([-1, 1, 1]) + torch.as_tensor([w, 0, 0])
        target["keypoints"] = scaled_keypoints

    if "flipped" in target:
        target["flipped"] = not target["flipped"]

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    new_transformation = _get_inverse_affine_matrix(
        center=[0, 0],
        angle=0,
        translate=[0, 0],
        scale=[ratio_width, ratio_height],
        shear=[0, 0],
    )

    new_transformation = torch.as_tensor([new_transformation + [0, 0, 1]])
    new_transformation = torch.linalg.inv(new_transformation.reshape(3, 3))

    if "transformation" in target:
        target_tranformation = target["transformation"]
    else:
        target_tranformation = torch.eye(3)

    target["transformation"] = new_transformation @ target_tranformation

    if "masks" in target:
        target["masks"] = interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding, fill=None):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]), fill=tuple(fill))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(target["masks"], (0, padding[0], 0, padding[1]), fill=fill)
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:

            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class PadToAspectRatio(object):
    def __init__(self, aspect_ratio=4 / 3):
        self.aspect_ratio = aspect_ratio

    def __call__(self, img, target):
        width, height = img.size
        max_width = math.ceil(max(width, self.aspect_ratio * height))
        max_height = math.ceil(max(height, width / self.aspect_ratio))

        return pad(img, target, (max_height - height, max_width - width))


class PadCropToSize(object):
    def __init__(self, size, fill=None):
        self.size = size
        self.fill = fill

    def __call__(self, img, target):
        width, height = img.size
        pad_value = (max(self.size[0] - width, 0), max(self.size[1] - height, 0))

        pad_img, pad_target = pad(img, target, pad_value, fill=self.fill)
        # i, j, h, w
        width, height = pad_img.size
        crop_width, crop_height = max(0, width - self.size[0]), max(0, height - self.size[1])
        region = (crop_height // 2, crop_width // 2, self.size[1], self.size[0])
        final_img, final_target = crop(pad_img, pad_target, region=region)
        # print(f"{img.size} {pad_img.size} {final_img.size} {region}")
        return final_img, final_target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms: List[object], p: List[float] = None):
        self.transforms = transforms
        if p is None:
            p = [1 / len(self.transforms) for x in self.transforms]
        self.p = p

    def __call__(self, img, target):
        return random.choices(self.transforms, weights=self.p)[0](img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = keypoints / torch.tensor([w, h, 1], dtype=torch.float32)
            target["keypoints"] = keypoints
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# Randaugment


def _get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: List[float], shear: List[float]
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    # matrix = [d, -b, 0.0, -c, a, 0.0]

    # scale
    matrix = [d / scale[0], -b / scale[1], 0.0, -c / scale[0], a / scale[1], 0.0]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def affine_transform(image, matrix, interpolation, fill):
    assert not isinstance(image, torch.Tensor), "Affine only works with PIL"

    pil_interpolation = pil_modes_mapping[interpolation]
    return F_pil.affine(image, matrix=matrix, interpolation=pil_interpolation, fill=fill)


class RandomAffine(object):
    def __init__(
        self,
        x=None,
        y=None,
        sx=None,
        sy=None,
        angle=None,
        scale_x=None,
        scale_y=None,
        interpolation="BILINEAR",
        fill=None,
    ):
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy
        self.angle = angle
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.interpolation = interpolation
        self.fill = fill

    def random_generator(self):
        result = {
            "x": 0.0,
            "y": 0.0,
            "sx": 0.0,
            "sy": 0.0,
            "angle": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
        }

        if self.sx is not None:
            if isinstance(self.sx, (list, tuple)):
                assert len(self.sx) == 2
                sx = np.random.random() * (self.sx[1] - self.sx[0]) + self.sx[0]
                result["sx"] = sx
            else:
                assert isinstance(self.sx, (int, float))
                result["sx"] = self.sx

        if self.sy is not None:
            if isinstance(self.sy, (list, tuple)):
                assert len(self.sy) == 2
                sy = np.random.random() * (self.sy[1] - self.sy[0]) + self.sy[0]
                result["sy"] = sy
            else:
                assert isinstance(self.sy, (int, float))
                result["sy"] = self.sy

        if self.angle is not None:
            if isinstance(self.angle, (list, tuple)):
                assert len(self.angle) == 2
                angle = np.random.random() * (self.angle[1] - self.angle[0]) + self.angle[0]
                result["angle"] = angle
            else:
                assert isinstance(self.angle, (int, float))
                result["angle"] = self.angle

        if self.x is not None:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                result["x"] = x
            else:
                assert isinstance(self.x, (int, float))
                result["x"] = self.x

        if self.y is not None:
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                result["y"] = y
            else:
                assert isinstance(self.y, (int, float))
                result["y"] = self.y

        if self.scale_x is not None:
            if isinstance(self.scale_x, (list, tuple)):
                assert len(self.scale_x) == 2
                scale_x = np.random.random() * (self.scale_x[1] - self.scale_x[0]) + self.scale_x[0]
                result["scale_x"] = scale_x
            else:
                assert isinstance(self.scale_x, (int, float))
                result["scale_x"] = self.scale_x

        if self.scale_y is not None:
            if isinstance(self.scale_y, (list, tuple)):
                assert len(self.scale_y) == 2
                scale_y = np.random.random() * (self.scale_y[1] - self.scale_y[0]) + self.scale_y[0]
                result["scale_y"] = scale_y
            else:
                assert isinstance(self.scale_y, (int, float))
                result["scale_y"] = self.scale_y

        return result

    def __call__(self, img, target):
        target = copy.deepcopy(target)
        # random.random() < self.p:
        parameter = self.random_generator()
        h = target["size"][0]
        w = target["size"][1]

        new_transformation = _get_inverse_affine_matrix(
            center=[h * 0.5, w * 0.5],
            angle=parameter["angle"],
            translate=[w * parameter["x"], h * parameter["y"]],
            scale=[parameter["scale_x"], parameter["scale_y"]],
            shear=[parameter["sx"], parameter["sy"]],
        )

        img = affine_transform(
            img,
            matrix=new_transformation,
            interpolation=self.interpolation,
            fill=self.fill,
        )

        new_transformation = torch.as_tensor([new_transformation + [0, 0, 1]])
        new_transformation = torch.linalg.inv(new_transformation.reshape(3, 3))

        if "transformation" in target:
            target_tranformation = target["transformation"]
        else:
            target_tranformation = torch.eye(3)

        target["transformation"] = new_transformation @ target_tranformation

        shift = torch.as_tensor([w * 0.5 + 0.5, h * 0.5 + 0.5])
        shift = torch.as_tensor([0, 0])

        # TODO make this better
        if "keypoints" in target:
            keypoints = target["keypoints"]
            for k in range(keypoints.shape[0]):
                for p in range(keypoints.shape[1]):
                    point = keypoints[k, p, :2] - shift
                    point = torch.cat([point, torch.ones([1])])

                    new_point = new_transformation @ point
                    new_point = new_point[:2] + shift
                    if new_point[0] < 0 or new_point[1] < 0 or new_point[0] >= w or new_point[1] >= h:
                        keypoints[k, p] = torch.as_tensor([0, 0, 0])
                    else:
                        keypoints[k, p, :2] = new_point
            target["keypoints"] = keypoints

        if "boxes" in target:
            boxes = target["boxes"]
            for k in range(boxes.shape[0]):
                # compute all 4 points of a box

                x_min = torch.min(boxes[k, ::2]) - shift[0]
                x_max = torch.max(boxes[k, ::2]) - shift[0]
                y_min = torch.min(boxes[k, 1::2]) - shift[1]
                y_max = torch.max(boxes[k, 1::2]) - shift[1]
                points = []

                points.append(torch.as_tensor([x_min, y_min, 1]))
                points.append(torch.as_tensor([x_max, y_min, 1]))
                points.append(torch.as_tensor([x_min, y_max, 1]))
                points.append(torch.as_tensor([x_max, y_max, 1]))

                new_points = []
                for p in points:
                    new_points.append(new_transformation @ p)
                new_points = torch.stack(new_points, axis=0)[:, :2]

                x_new_point_min = torch.min(new_points[:, ::2]) + shift[0]
                x_new_point_max = torch.max(new_points[:, ::2]) + shift[0]
                y_new_point_min = torch.min(new_points[:, 1::2]) + shift[1]
                y_new_point_max = torch.max(new_points[:, 1::2]) + shift[1]

                new_box = torch.as_tensor([x_new_point_min, y_new_point_min, x_new_point_max, y_new_point_max])

                # clip bbox
                max_size = torch.as_tensor([w, h], dtype=torch.float32)

                new_box = torch.min(new_box.reshape(2, 2), max_size)
                new_box = new_box.clamp(min=0)
                area = (new_box[1, :] - new_box[0, :]).prod()
                target["boxes"][k] = new_box.reshape(4)
                target["area"][k] = area

        return img, target


class Identity:
    def __call__(self, img, target):
        return img, target


class AutoContrast:
    def __init__(self):
        pass

    def __call__(self, img, target):
        img = PIL.ImageOps.autocontrast(img)
        return img, target


class RandomEqualize:
    def __init__(self):
        pass

    def __call__(self, img, target):
        img = PIL.ImageOps.equalize(img)
        return img, target


class RandomSolarize:
    def __init__(self, magnitude=10):
        self.magnitude = magnitude
        self.range = np.linspace(256, 231, 10)

    def __call__(self, img, target):
        magnitude = np.random.randint(0, self.magnitude)
        img = PIL.ImageOps.solarize(img, magnitude)
        return img, target


class RandomColor:
    def __init__(self, magnitude=10):
        self.magnitude = magnitude
        self.range = np.linspace(0.0, 0.9, 10)

    def __call__(self, img, target):
        magnitude = np.random.randint(0, self.magnitude)
        PIL.ImageEnhance.Color(img).enhance(1 + self.range[magnitude] * random.choice([-1, 1]))
        return img, target


class RandomContrast:
    def __init__(self, magnitude=10):
        self.magnitude = magnitude
        self.range = np.linspace(0.0, 0.5, 10)

    def __call__(self, img, target):
        magnitude = np.random.randint(0, self.magnitude)
        PIL.ImageEnhance.Contrast(img).enhance(1 + self.range[magnitude] * random.choice([-1, 1]))
        return img, target


class RandomBrightness:
    def __init__(self, magnitude=10):
        self.magnitude = magnitude
        self.range = np.linspace(0.0, 0.3, 10)

    def __call__(self, img, target):
        magnitude = np.random.randint(0, self.magnitude)
        PIL.ImageEnhance.Brightness(img).enhance(1 + self.range[magnitude] * random.choice([-1, 1]))
        return img, target


class RandomSharpness:
    def __init__(self, magnitude=10):
        self.magnitude = magnitude
        self.range = np.linspace(0.0, 0.9, 10)

    def __call__(self, img, target):

        magnitude = np.random.randint(0, self.magnitude)
        PIL.ImageEnhance.Sharpness(img).enhance(1 + self.range[magnitude] * random.choice([-1, 1]))
        return img, target


class RandomPosterize:
    def __init__(self, magnitude=10):
        self.magnitude = magnitude
        self.range = np.round(np.linspace(8, 4, 10), 0).astype(np.int)

    def __call__(self, img, target):
        magnitude = np.random.randint(0, self.magnitude)
        img = PIL.ImageOps.posterize(img, self.range[magnitude])
        return img, target
