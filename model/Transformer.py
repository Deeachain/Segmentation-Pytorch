import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.base_model import build_backbone
from model.trans2seg.transformer import VisionTransformer
from model.trans2seg1.transformer import VisionTransformer


# class Transformer(nn.Module):
#     def __init__(self, c4_channels=2048, embed_dim=512, depth=6, num_heads=4, mlp_ratio=4., feat_HxW=1024, nclass=6):
#         super().__init__()
#         self.vit = VisionTransformer(input_dim=c4_channels,
#                                      embed_dim=embed_dim,
#                                      depth=depth,
#                                      num_heads=num_heads,
#                                      mlp_ratio=mlp_ratio,
#                                      num_patches=feat_HxW,
#                                      feat_HxW=feat_HxW,
#                                      nclass=nclass,
#                                      drop_rate=1.0,
#                                      attn_drop_rate=1.0,
#                                      drop_path_rate=1.0)

#     def forward(self, x):
#         n, _, h, w = x.shape
#         x = self.vit.hybrid_embed(x)
#         cls_token, encoder = self.vit.forward_encoder(x)
#         decoder_list = self.vit.forward_decoder(encoder)
#         encoder = encoder.reshape(n, h, w, -1).permute(0, 3, 1, 2)
#         return encoder, decoder_list  # x is encoder output [B,C,H,W], attns_list is decoder q*k

class Transformer(nn.Module):
    def __init__(self, c4_channels=2048):
        super().__init__()
        last_channels = 256
        self.vit = VisionTransformer(input_dim=c4_channels,
                                     embed_dim=last_channels,
                                     depth=8,
                                     num_heads=4,
                                     mlp_ratio=4.0,
                                     decoder_feat_HxW=32*32)

    def forward(self, x):
        n, _, h, w = x.shape
        x = self.vit.hybrid_embed(x)

        cls_token, x = self.vit.forward_encoder(x)

        attns_list = self.vit.forward_decoder(x)

        x = x.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return x, attns_list

def _expand(x, nclass):
    return x.unsqueeze(1).repeat(1, nclass, 1, 1, 1).flatten(0, 1)


class SegTrans(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', out_stride=32, mult_grid=False):
        super(SegTrans, self).__init__()

        in_channels = 512
        num_heads = 4
        depth = 8
        embed_dim = 512
        feat_HxW = 16 * 16
        if backbone == 'resnet18' or backbone == 'resnet34':
            expansion = 1
        elif backbone == 'resnet50' or backbone == 'resnet101':
            expansion = 4
        self.num_classes = num_classes
        self.backbone = build_backbone(backbone, out_stride, mult_grid)

        # self.transformer = Transformer(c4_channels=in_channels * expansion,
        #                                embed_dim=embed_dim,
        #                                depth=depth,
        #                                num_heads=num_heads,
        #                                mlp_ratio=4.,
        #                                feat_HxW=feat_HxW,
        #                                nclass=num_classes)

        self.transformer = Transformer(c4_channels=in_channels * expansion)

        # self.lay1 = nn.Linear(num_classes * num_heads, num_classes)
        # self.conv_1 = nn.Conv2d(512 * expansion, num_classes, 1)
        self.conv_2 = nn.Conv2d(64 * expansion, num_classes, 1)


        self.lay1 = nn.Sequential(nn.Conv2d(256+num_heads, embed_dim, 3))
        self.lay2 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3))
        self.lay3 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3))

        self.pred = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        layers = self.backbone(x)  # resnet 4 layers
        # encoder, decoder_list = self.transformer(layers[3])  # encoder output,

        feat_enc, attns_list = self.transformer(layers[3])
        # print(feat_enc.shape, attns_list[-1].shape)
        attn_map = attns_list[-1]
        B, nclass, nhead, _ = attn_map.shape
        _, _, H, W = feat_enc.shape
        attn_map = attn_map.reshape(B*nclass, nhead, H, W)

        x = torch.cat([_expand(feat_enc, nclass), attn_map], 1)
        x = self.lay1(x)
        x = self.lay2(x)

        size = layers[0].size()[2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        layers0 = self.conv_2(layers[0])
        x = x + _expand(layers0, nclass)

        x = self.lay3(x)
        x = self.pred(x).reshape(B, nclass, size[0], size[1])

        # print(self.lay1(attn_map).shape)
        # trans = self.lay1(attn_map).permute(0, 2, 1).reshape(B, self.num_classes, encoder.shape[-2],
        #                                                              encoder.shape[-1])

        # # Upsampling
        # layers3 = self.conv_1(layers[3])
        # temp = trans + layers3

        # temp = F.interpolate(temp, layers[0].size()[2:], mode="bilinear", align_corners=True)
        # # temp = F.interpolate(trans, layers[0].size()[2:], mode="bilinear", align_corners=True)
        # layers0 = self.conv_2(layers[0])
        # output = temp + layers0
        # output = F.interpolate(output, (H, W), mode="bilinear", align_corners=True)

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
