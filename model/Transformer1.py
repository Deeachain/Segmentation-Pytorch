from model.trans2seg1.transformer import VisionTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.base_model import build_backbone


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Transformer(nn.Module):
    def __init__(self, layer3_channels=2048, embed_dim=512, depth=8, num_heads=6, mlp_ratio=1.0, feat_HxW=16*16, num_classes=6):
        super().__init__()
        self.vit = VisionTransformer(input_dim=layer3_channels, embed_dim=embed_dim, depth=depth, 
                                     num_patches=feat_HxW, num_classes=num_classes, feat_HxW=feat_HxW, 
                                     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
                                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm)

    def forward(self, x):
        n, _, h, w = x.shape
        x = self.vit.hybrid_embed(x)

        cls_token, x = self.vit.forward_encoder(x)

        attns_list = self.vit.forward_decoder(x)

        x = x.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return x, attns_list


class TransformerHead(nn.Module):
    def __init__(self, layer0_channels=256, layer3_channels=2048, hid_dim=64, embed_dim=512, depth=8, 
                 num_heads=8, mlp_ratio=1.0, feat_HxW=16*16, num_classes=6, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.transformer = Transformer(layer3_channels, embed_dim, depth, num_heads, mlp_ratio, feat_HxW, num_classes)

        self.conv_c1 = _ConvBNReLU(layer0_channels, hid_dim, 1, norm_layer=norm_layer)

        self.lay1 = SeparableConv2d(embed_dim+num_heads, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay2 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay3 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)

        self.pred = nn.Conv2d(hid_dim, 1, 1)


    def forward(self, x, c1):
        feat_enc, attns_list = self.transformer(x)
        attn_map = attns_list[-1]
        B, num_classes, nhead, _ = attn_map.shape
        _, _, H, W = feat_enc.shape
        attn_map = attn_map.reshape(B*num_classes, nhead, H, W)

        x = torch.cat([_expand(feat_enc, num_classes), attn_map], 1)

        x = self.lay1(x)
        x = self.lay2(x)

        size = c1.size()[2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        c1 = self.conv_c1(c1)
        x = x + _expand(c1, num_classes)

        x = self.lay3(x)
        x = self.pred(x).reshape(B, num_classes, size[0], size[1])

        return x

def _expand(x, num_classes):
    return x.unsqueeze(1).repeat(1, num_classes, 1, 1, 1).flatten(0, 1)



class SegTrans(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', out_stride=32, mult_grid=False):
        super(SegTrans, self).__init__()

        layer0_channels = 64
        layer3_channels = 512
        num_heads = 4
        depth = 8
        embed_dim = 512
        hid_dim= 64
        feat_HxW = 32 * 32
        mlp_ratio = 4.0
        if backbone == 'resnet18' or backbone == 'resnet34':
            expansion = 1
        elif backbone == 'resnet50' or backbone == 'resnet101':
            expansion = 4
        self.num_classes = num_classes
        self.backbone = build_backbone(backbone, out_stride, mult_grid)

        self.transformer_head = TransformerHead(layer0_channels=layer0_channels*expansion, 
                                                layer3_channels=layer3_channels*expansion, 
                                                hid_dim=hid_dim, embed_dim=embed_dim, 
                                                depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                                feat_HxW=feat_HxW, num_classes=num_classes, norm_layer=nn.BatchNorm2d)


    def forward(self, x):
        B, C, H, W = x.shape
        layers = self.backbone(x)  # resnet 4 layers
        x = self.transformer_head(layers[3], layers[0])
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x


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
        modules = [self.transformer, self.lay1, self.conv_1, self.conv_2]
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

