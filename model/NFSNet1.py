# _*_ coding: utf-8 _*_
"""
Time:     2020/12/31 下午7:38
Author:   Ding Cheng(Deeachain)
File:     NFSNet1.py
Github:   https://github.com/Deeachain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.base_model.resnet import resnet18, resnet34, resnet50


class sat(nn.Module):
    def __init__(self, in_channels, output_channels, feature_h, feature_w):
        super(sat, self).__init__()
        self.output_channels = output_channels
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.query = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=self.output_channels, kernel_size=1),
                                   nn.BatchNorm2d(self.output_channels), nn.ReLU())
        self.key = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=self.output_channels, kernel_size=1),
                                 nn.BatchNorm2d(self.output_channels), nn.ReLU())
        self.value = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=self.output_channels, kernel_size=1),
                                   nn.BatchNorm2d(self.output_channels), nn.ReLU())
        self.softmax = nn.Softmax(dim=1)
        self.conv_last = nn.Sequential(nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels,
                                                 kernel_size=3, padding=1, groups=self.output_channels),
                                       nn.BatchNorm2d(self.output_channels))

    def forward(self, x):
        B, C, H, W = x.size()

        query = self.query(x)
        _, c, h, w = query.size()
        query = query.view(B, c, -1)  # b,c,feature_h*feature_w

        key = self.key(x)
        _, c, h, w = key.size()
        key = key.view(B, c, -1).transpose(2, 1)  # b,c,feature_h*feature_w

        value = self.value(x)
        _, c, h, w = value.size()
        value = value.view(B, c, -1)  # b,c,feature_h*feature_w

        temp = torch.bmm(query, key)
        temp = self.softmax(temp)
        transformer_layer = torch.bmm(temp, value)

        transformer_layer = transformer_layer.view(B, self.output_channels, self.feature_h,
                                                   self.feature_w)  # b,self.output_channels,feature_h, feature_w
        transformer_layer = self.conv_last(transformer_layer)

        return transformer_layer


class NFSNet1(nn.Module):
    def __init__(self, num_classes, backbone=resnet18):
        super(NFSNet1, self).__init__()

        self.backbone = backbone(pretrained=False)
        if backbone == resnet18 or backbone == resnet34:
            in_channels = 512
            output_channels = 256
        else:
            in_channels = 2048
            output_channels = 1024

        h, w = 16, 16
        self.sat1 = sat(in_channels=in_channels, output_channels=output_channels,
                                                    feature_h=h, feature_w=w)
        self.sat2 = sat(in_channels=in_channels // 2, output_channels=output_channels // 2,
                                                    feature_h=h * 2, feature_w=w * 2)
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=1),
                                          nn.BatchNorm2d(output_channels))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels // 2, out_channels=output_channels // 2, kernel_size=1),
                                          nn.BatchNorm2d(output_channels // 2))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=output_channels // 2, kernel_size=1),
                                          nn.BatchNorm2d(output_channels // 2))
        self.conv_last = nn.Sequential(nn.Conv2d(in_channels=output_channels // 2, out_channels=num_classes, kernel_size=1),
                                       nn.BatchNorm2d(num_classes), nn.ReLU())
        self.relu = nn.ReLU()

    def forward(self, x):
        layers = self.backbone(x)  # resnet 4 blocks

        output1 = self.sat1(layers[-1])
        output2 = self.sat2(layers[-2])

        output1 = self.conv_2(output1)
        output1 = F.interpolate(output1, layers[-2].size()[2:], mode="bilinear", align_corners=True)
        layers_last2 = self.conv_3(layers[-1])
        layers_last2 = F.interpolate(layers_last2, layers[-2].size()[2:], mode="bilinear", align_corners=True)
        output = torch.add(torch.add(output1, output2), layers_last2)

        output = self.conv_last(output)
        output = F.interpolate(output, x.size()[2:], mode="bilinear", align_corners=True)

        return output


"""print layers and params of network"""
if __name__ == '__main__':
    model = NFSNet1(num_classes=3, backbone=resnet18)
    summary(model, (3, 512, 512), device="cpu")
