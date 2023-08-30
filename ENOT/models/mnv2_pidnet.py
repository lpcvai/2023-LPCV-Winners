from collections import OrderedDict
import logging
import types
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from torchvision.models import MobileNet_V2_Weights
from torchvision.models.mobilenetv2 import InvertedResidual

from .model_utils import segmenthead, PAPPM, PagFM, Light_Bag


branch1_output = None
branch2_output = None
branch3_output = None


def _branch1_forward_hook(m, inputs, output):
    global branch1_output
    branch1_output = output
    return output


def _branch2_forward_hook(m, inputs, output):
    global branch2_output
    branch2_output = output
    return output


def _branch3_forward_hook(m, inputs, output):
    global branch3_output
    branch3_output = output
    return output


def _mobilenet_forward_wrapper(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    return x


def construct_mobilenet_v2():
    mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
    mobilenet_v2.classifier = nn.Identity()

    funcType = types.MethodType
    mobilenet_v2._forward_impl = funcType(_mobilenet_forward_wrapper, mobilenet_v2)
    return mobilenet_v2


class MobilenetV2_PIDNet(nn.Module):
    def __init__(self, num_classes=14, augment=True):
        super(MobilenetV2_PIDNet, self).__init__()

        self.augment = augment
        if self.augment:
            self.aux_head_p = segmenthead(32, 128, num_classes)
            self.aux_head_d = segmenthead(32, 32, 1)

        self.backbone = construct_mobilenet_v2()
        self.ppm = PAPPM(inplanes=1280, branch_planes=96, outplanes=64)
        self.bag = Light_Bag(in_channels=64, out_channels=64)
        self.final_layer = segmenthead(64, 128, num_classes)

        # Branch D
        self.layer1_d = self.make_branch_block(inp=32, oup=32, expand_ratio=6)
        self.layer2_d = self.make_branch_block(inp=32, oup=32, expand_ratio=6)
        self.layer3_d = self.make_branch_block(inp=32, oup=64, expand_ratio=6)

        self.compress1_d = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.compress2_d = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )

        # Branch P
        self.layer1_p = self.make_branch_block(inp=32, oup=32, expand_ratio=6)
        self.layer2_p = self.make_branch_block(inp=32, oup=32, expand_ratio=6)
        self.layer3_p = self.make_branch_block(inp=32, oup=64, expand_ratio=6)

        self.compress1_p = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.compress2_p = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )

        self.pag1 = PagFM(32, 32)
        self.pag2 = PagFM(32, 32)

        self._register_branch_hooks()


    def make_branch_block(self, inp=32, oup=32, expand_ratio=6):
        layer = nn.Sequential(
            InvertedResidual(inp=inp, oup=oup, stride=1, expand_ratio=expand_ratio),
            InvertedResidual(inp=oup, oup=oup, stride=1, expand_ratio=expand_ratio),
        )
        return layer


    def _register_branch_hooks(self):
        self.backbone.features[4].register_forward_hook(_branch1_forward_hook)
        self.backbone.features[7].register_forward_hook(_branch2_forward_hook)
        self.backbone.features[14].register_forward_hook(_branch3_forward_hook)


    def forward(self, x):
        height, width = x.shape[-2] // 8, x.shape[-1] // 8

        x = self.backbone(x)

        # Stage 1
        x_p = self.layer1_p(branch1_output)
        x_p = self.pag1(x_p, self.compress1_p(branch2_output))
        if self.augment:
            temp_p = x_p

        x_d = self.layer1_d(branch1_output)
        x_d = x_d + F.interpolate(self.compress1_d(branch2_output), size=(height, width), mode="bilinear")

        # Stage 2
        x_p = self.layer2_p(x_p)
        x_p = self.pag2(x_p, self.compress2_p(branch3_output))

        x_d = self.layer2_d(x_d)
        x_d = x_d + F.interpolate(self.compress2_d(branch3_output), size=(height, width), mode="bilinear")
        if self.augment:
            temp_d = x_d

        # Stage 3
        x_p = self.layer3_p(x_p)
        x_d = self.layer3_d(x_d)
        x = self.ppm(x)
        x = F.interpolate(x, size=(height, width), mode="bilinear")

        # Fusing
        x = self.bag(x_p, x, x_d)
        x = self.final_layer(x)

        if self.augment:
            aux_p = self.aux_head_p(temp_p)
            aux_d = self.aux_head_d(temp_d)
            return [aux_p, x, aux_d]
        return x
