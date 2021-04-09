import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from model.Transformer import Transformer
from model.trans2seg.basic import SeparableConv2d
from model.base_model.resnet import resnet18, resnet34, resnet50

# from .sync_bn.inplace_abn.bn import InPlaceABNSync

from torchsummary import summary

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class MR_Transformer(nn.Module):
    def __init__(self, backbone=resnet18, num_classes=6):
        super(MR_Transformer, self).__init__()

        # mult path
        mult_path = []
        mult_path.append(self._make_layer(BasicBlock, 3, 64, 1, stride=2))
        mult_path.append(self._make_layer(BasicBlock, 64, 64, 1, stride=1))
        mult_path.append(self._make_layer(BasicBlock, 64, 64, 1, stride=1))
        mult_path.append(self._make_layer(BasicBlock, 64, 128, 1, stride=2))
        mult_path.append(self._make_layer(BasicBlock, 128, 128, 1, stride=1))
        mult_path.append(self._make_layer(BasicBlock, 128, 128, 1, stride=1))
        mult_path.append(self._make_layer(BasicBlock, 128, 256, 1, stride=2))
        mult_path.append(self._make_layer(BasicBlock, 256, 256, 1, stride=1))
        mult_path.append(self._make_layer(BasicBlock, 256, 256, 1, stride=1))
        self.MR = nn.ModuleList(mult_path)
        self.class_conv = nn.Conv2d(256, num_classes, 1)

        # transformer path
        self.backbone = backbone(pretrained=False)
        self.num_classes = num_classes
        layer0_dim = 64
        num_heads = 4
        depth = 12
        embed_dim = 256
        feat_HxW = 16*16
        if backbone == resnet18 or backbone == resnet34:
            in_channels = 512
        else:
            in_channels = 2048
            layer0_dim *= 4

        self.transformer = Transformer(c4_channels=in_channels,
                                       embed_dim=embed_dim,
                                       depth=depth,
                                       num_heads=num_heads,
                                       mlp_ratio=1.,
                                       feat_HxW=feat_HxW,
                                       nclass=num_classes)
        self.lay1 = nn.Linear(num_classes * num_heads, num_classes)
        self.layers3_conv = nn.Conv2d(in_channels, num_classes, 1)
        self.layers2_conv = nn.Conv2d(in_channels // 2, num_classes, 1)
        self.layers1_conv = nn.Conv2d(in_channels // 4, num_classes, 1)
        self.layers0_conv = nn.Conv2d(in_channels // 8, num_classes, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        B, _, H, W = input.shape

        # Mult path
        mr = x
        for block in self.MR:
            mr = block(mr)

        # Transformer path
        layers = self.backbone(input)  # resnet 4 layers
        encoder, decoder_list = self.transformer(layers[3])  # encoder output,
        trans = self.lay1(decoder_list[-1]).permute(0, 2, 1).reshape(B, self.num_classes, encoder.shape[-2], encoder.shape[-1])

        # Upsampling
        mr = self.class_conv(mr)
        mr = F.interpolate(mr, (H, W), mode='bilinear', align_corners=True)
        layers3 = self.layers3_conv(layers[3])
        trans = trans + layers3

        trans = F.interpolate(trans, (layers[2].shape[-2], layers[2].shape[-1]), mode='bilinear', align_corners=True)
        layers2 = self.layers2_conv(layers[2])
        trans = trans + layers2

        trans = F.interpolate(trans, (layers[1].shape[-2], layers[1].shape[-1]), mode='bilinear', align_corners=True)
        layers1 = self.layers1_conv(layers[1])
        trans = trans + layers1

        trans = F.interpolate(trans, (layers[0].shape[-2], layers[0].shape[-1]), mode='bilinear', align_corners=True)
        layers0 = self.layers0_conv(layers[0])
        trans = trans + layers0
        trans = F.interpolate(trans, (H, W), mode='bilinear', align_corners=True)

        output = mr + trans
        # output = trans
        return output



if __name__ == '__main__':
    model = MR_Transformer(backbone=resnet18, num_classes=6)
    summary(model, (3, 512, 512), device="cpu")