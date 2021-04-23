# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 19:27
Author:   Ding Cheng(Deeachain)
File:     BiSeNet.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.base_model import build_backbone


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                       has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                       has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        return output


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class BiSeNet(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d, backbone='resnet18'):
        super(BiSeNet, self).__init__()
        conv_channel = 128
        self.spatial_path = SpatialPath(3, conv_channel, norm_layer)

        if backbone == 'resnet18' or backbone == 'resnet34':
            expansion = 1
        elif backbone == 'resnet50' or backbone == 'resnet101':
            expansion = 4

        self.backbone = build_backbone(backbone)

        # resnet layers < 50  stage = [512, 256, 128, 64]; resnet layers > 50  stage = [2048, 1024, 512, 256]
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(512 * expansion, conv_channel, 1, 1, 0,
                                                                                has_bn=True, has_relu=True,
                                                                                has_bias=False, norm_layer=norm_layer))

        self.arms1 = AttentionRefinement(512 * expansion, conv_channel, norm_layer)
        self.arms2 = AttentionRefinement(256 * expansion, conv_channel, norm_layer)
        self.refines = ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                                  has_bn=True, norm_layer=norm_layer,
                                  has_relu=True, has_bias=False)

        self.heads1 = BiSeNetHead(conv_channel * 2, num_classes, False, norm_layer)
        self.heads2 = BiSeNetHead(conv_channel, num_classes, True, norm_layer)
        self.heads3 = BiSeNetHead(conv_channel, num_classes, True, norm_layer)

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1, norm_layer)

        self._init_weight()

    def forward(self, x):
        size = x.shape
        spatial_out = self.spatial_path(x)

        context_blocks = self.backbone(x)

        global_context = self.global_context(context_blocks[-1])  # change channel
        global_context = F.interpolate(global_context, size=context_blocks[-1].size()[2:], mode='bilinear',
                                       align_corners=True)
        last_fm = global_context

        arm1 = self.arms1(context_blocks[-1])
        arm1 += last_fm
        arm1 = F.interpolate(arm1, size=(context_blocks[-2].size()[2:]), mode='bilinear', align_corners=True)
        arm1 = self.refines(arm1)
        last_fm = arm1

        arm2 = self.arms2(context_blocks[-2])
        arm2 += last_fm
        arm2 = F.interpolate(arm2, size=(context_blocks[-3].size()[2:]), mode='bilinear', align_corners=True)
        arm2 = self.refines(arm2)
        context_out = arm2

        concate_fm = self.ffm(spatial_out, context_out)

        main = self.heads1(concate_fm)
        aux_0 = self.heads2(arm2)
        aux_1 = self.heads3(arm1)
        main = F.interpolate(main, size=size[2:], mode='bilinear', align_corners=True)
        aux_0 = F.interpolate(aux_0, size=size[2:], mode='bilinear', align_corners=True)
        aux_1 = F.interpolate(aux_1, size=size[2:], mode='bilinear', align_corners=True)

        return main, aux_0, aux_1

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
        modules = [self.global_context, self.arms1, self.arms2, self.refines, self.heads1, self.heads2, self.heads3, self.ffm]
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
    model = BiSeNet(num_classes=3, backbone='resnet18')
    print(model)
    # summary(model, (3, 512, 512), device="cpu")
