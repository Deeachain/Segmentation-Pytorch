# _*_ coding: utf-8 _*_
"""
Time:     2021/1/11 下午10:22
Author:   Ding Cheng(Deeachain)
File:     NFSNet_ppm.py
Github:   https://github.com/Deeachain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.base_model.resnet import resnet18, resnet34, resnet50


class sat(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(sat, self).__init__()
        self.output_channels = output_channels
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
        sat = torch.bmm(temp, value)

        sat = sat.view(B, self.output_channels, H, W)  # b,self.output_channels,feature_h, feature_w
        sat = self.conv_last(sat)

        return sat


# without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.global_avgpool(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class NFSNet(nn.Module):
    def __init__(self, num_classes, backbone=resnet18):
        super(NFSNet, self).__init__()

        self.backbone = backbone(pretrained=False)
        if backbone == resnet18 or backbone == resnet34:
            in_channels = 512
            output_channels = 256
        else:
            in_channels = 2048
            output_channels = 1024


        self.PPM1 = PPM(in_dim=in_channels, reduction_dim=in_channels // 4, bins=[1,2,3,6])
        self.PPM2 = PPM(in_dim=in_channels // 2, reduction_dim=(in_channels // 2) // 4, bins=[1,2,3,6])

        self.sat1 = sat(in_channels=in_channels + in_channels, output_channels=output_channels)
        self.sat2 = sat(in_channels=in_channels // 2 + in_channels // 2 , output_channels=output_channels // 2)


        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=1),
                                          nn.BatchNorm2d(output_channels))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels // 2, out_channels=output_channels // 2, kernel_size=1),
                                          nn.BatchNorm2d(output_channels // 2))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=output_channels // 2, kernel_size=1),
                                          nn.BatchNorm2d(output_channels // 2))
        self.conv_last = nn.Sequential(nn.Conv2d(in_channels=output_channels // 2, out_channels=num_classes, kernel_size=1),
                                       nn.BatchNorm2d(num_classes), nn.ReLU())



    def forward(self, x):
        layers = self.backbone(x)  # resnet 4 blocks

        layers_4 = self.PPM1(layers[-1])
        layers_3 = self.PPM2(layers[-2])

        sat1 = self.sat1(layers_4)
        sat2 = self.sat2(layers_3)


        # GFR layers
        sat_avg1 = self.global_avgpool(sat1)
        layers_last1 = self.conv_1(layers[-1])
        output1 = torch.add(torch.mul(sat_avg1, layers_last1), sat1)
        output1 = torch.add(layers_last1, output1)

        sat_avg2 = self.global_avgpool(sat2)
        layers_last_by_one = self.conv_2(layers[-2])
        output2 = torch.add(torch.mul(sat_avg2, layers_last_by_one), sat2)
        output2 = torch.add(layers_last_by_one, output2)


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
    model = NFSNet(num_classes=3, backbone=resnet18)
    summary(model, (3, 512, 512), device="cpu")
