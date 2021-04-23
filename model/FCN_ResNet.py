# _*_ coding: utf-8 _*_
"""
Time:     2020/12/1 18:23
Author:   Ding Cheng(Deeachain)
File:     FCN_ResNet.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
# """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.base_model import build_backbone


class FCN_ResNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', out_stride=32, mult_grid=False):
        super(FCN_ResNet, self).__init__()

        if backbone == 'resnet18' or backbone == 'resnet34':
            expansion = 1
        elif backbone == 'resnet50' or backbone == 'resnet101':
            expansion = 4
        self.backbone = build_backbone(backbone, out_stride, mult_grid)

        self.conv_1 = nn.Conv2d(in_channels=512 * expansion, out_channels=num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=256 * expansion, out_channels=num_classes, kernel_size=1)
        self.conv_3 = nn.Conv2d(in_channels=128 * expansion, out_channels=num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(in_channels=64 * expansion, out_channels=num_classes, kernel_size=1)

        self._init_weight()

    def forward(self, x):
        layers = self.backbone(x)  # resnet 4 layers

        layers3 = self.conv_1(layers[3])
        layers3 = F.interpolate(layers3, layers[2].size()[2:], mode="bilinear", align_corners=True)
        layers2 = self.conv_2(layers[2])

        output = layers2 + layers3
        output = F.interpolate(output, layers[1].size()[2:], mode="bilinear", align_corners=True)
        layers1 = self.conv_3(layers[1])

        output = output + layers1
        output = F.interpolate(output, layers[0].size()[2:], mode="bilinear", align_corners=True)
        layers0 = self.conv_4(layers[0])

        output = output + layers0
        output = F.interpolate(output, x.size()[2:], mode="bilinear", align_corners=True)
        aux1 = F.interpolate(layers2, x.size()[2:], mode="bilinear", align_corners=True)

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.conv_1, self.conv_2, self.conv_3, self.conv_4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


"""print layers and params of network"""
if __name__ == '__main__':
    model = FCN_ResNet(num_classes=3, backbone='resnet18')
    summary(model, (3, 512, 512), device="cpu")
