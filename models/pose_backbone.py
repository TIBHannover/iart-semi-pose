import logging
import types

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from models.hrnet import HighResolutionNet
from models.position_encoding import build_position_encoding

from utils.misc import NestedTensor

logger = logging.getLogger(__name__)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)

        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class ResNetBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self, name: str, train_backbone: bool, return_interm_layers: bool, pretrained: bool, dilation: bool = False
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation], pretrained=pretrained, norm_layer=FrozenBatchNorm2d
        )
        logger.info(f"=> Loading backbone, pretrained: {pretrained}")
        assert name in ["resnet50", "resnet101"], "Number of channels is hard-coded"
        num_channels = 2048
        super().__init__(backbone, train_backbone, return_interm_layers)
        self.num_channels = num_channels
        if return_interm_layers:
            self.num_channels = [512, 1024, 2048]
        else:
            self.num_channels = [2048]


class HRNetBackbone(nn.Module):
    def __init__(self, cfg, return_interm_layers: bool, pretrained: bool = False):
        super().__init__()
        if return_interm_layers:
            raise NotImplementedError("HRNet backbone does not support return interm layers")
        else:
            self.num_channels = [2048]
        self.body = HighResolutionNet(cfg)
        # if pretrained:
        #     self.body.init_weights(cfg.MODEL.PRETRAINED)

    def forward(self, tensor_list):
        y = self.body(tensor_list.tensors)

        out = {}
        for name, x in enumerate(y):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(arch="resnet", pretrained=False):
    if arch.startswith("resnet"):

        cfg = types.SimpleNamespace()
        cfg.hidden_dim = 256
        cfg.position_embedding = "sine"
        position_embedding = build_position_encoding(cfg)
        backbone = ResNetBackbone(
            arch,
            train_backbone=True,
            return_interm_layers=False,
            pretrained=pretrained,
            dilation=False,
        )
    elif arch.startswith("hrnet"):
        cfg = types.SimpleNamespace()
        cfg.STAGE1 = types.SimpleNamespace()
        cfg.STAGE1.NUM_MODULES = 1
        cfg.STAGE1.NUM_RANCHES = 1
        cfg.STAGE1.BLOCK = "BOTTLENECK"
        cfg.STAGE1.NUM_BLOCKS = [4]
        cfg.STAGE1.NUM_CHANNELS = [64]
        cfg.STAGE1.FUSE_METHOD = "SUM"

        cfg.STAGE2 = types.SimpleNamespace()
        cfg.STAGE2.NUM_MODULES = 1
        cfg.STAGE2.NUM_BRANCHES = 2
        cfg.STAGE2.BLOCK = "BASIC"
        cfg.STAGE2.NUM_BLOCKS = [4, 4]
        cfg.STAGE2.NUM_CHANNELS = [32, 64]
        cfg.STAGE2.FUSE_METHOD = "SUM"

        cfg.STAGE3 = types.SimpleNamespace()
        cfg.STAGE3.NUM_MODULES = 4
        cfg.STAGE3.NUM_BRANCHES = 3
        cfg.STAGE3.BLOCK = "BASIC"
        cfg.STAGE3.NUM_BLOCKS = [4, 4, 4]
        cfg.STAGE3.NUM_CHANNELS = [32, 64, 128]
        cfg.STAGE3.FUSE_METHOD = "SUM"

        cfg.STAGE4 = types.SimpleNamespace()
        cfg.STAGE4.NUM_MODULES = 3
        cfg.STAGE4.NUM_BRANCHES = 4
        cfg.STAGE4.BLOCK = "BASIC"
        cfg.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        cfg.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
        cfg.STAGE4.FUSE_METHOD = "SUM"
        backbone = HRNetBackbone(cfg, pretrained=pretrained, return_interm_layers=False)

        cfg = types.SimpleNamespace()
        cfg.hidden_dim = 256
        cfg.position_embedding = "sine"
        position_embedding = build_position_encoding(cfg)

    else:
        raise NotImplementedError(f"Unsupported backbone type: {arch}")
    model = Joiner(backbone, position_embedding)
    # model.num_channels = 256
    return model
