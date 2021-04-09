import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from model.trans2seg.basic import _ConvBNReLU, SeparableConv2d
from model.trans2seg.module import _ASPP, _FCNHead
from ..config import cfg
from model.trans2seg.transformer import VisionTransformer
# from IPython import embed


__all__ = ['Trans2Seg']


class Trans2Seg(SegBaseModel):

    def __init__(self):
        super().__init__()
        if self.backbone.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        else:
            c1_channels = 256
            c4_channels = 2048

        vit_params = cfg.MODEL.TRANS2Seg
        hid_dim = cfg.MODEL.TRANS2Seg.hid_dim

        assert cfg.AUG.CROP == False and cfg.TRAIN.CROP_SIZE[0] == cfg.TRAIN.CROP_SIZE[1]\
               == cfg.TRAIN.BASE_SIZE == cfg.TEST.CROP_SIZE[0] == cfg.TEST.CROP_SIZE[1]
        c4_HxW = (cfg.TRAIN.BASE_SIZE // 16) ** 2

        vit_params['decoder_feat_HxW'] = c4_HxW

        self.transformer_head = TransformerHead(vit_params, c1_channels=c1_channels, c4_channels=c4_channels, hid_dim=hid_dim)
        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['transformer_head', 'auxlayer'] if self.aux else ['transformer_head'])


    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)

        x = self.transformer_head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class Transformer(nn.Module):
    def __init__(self, vit_params, c4_channels=2048):
        super().__init__()
        last_channels = vit_params['embed_dim']
        self.vit = VisionTransformer(input_dim=c4_channels,
                                     embed_dim=last_channels,
                                     depth=vit_params['depth'],
                                     num_heads=vit_params['num_heads'],
                                     mlp_ratio=vit_params['mlp_ratio'],
                                     decoder_feat_HxW=vit_params['decoder_feat_HxW'])

    def forward(self, x):
        n, _, h, w = x.shape
        x = self.vit.hybrid_embed(x)

        cls_token, x = self.vit.forward_encoder(x)

        attns_list = self.vit.forward_decoder(x)

        x = x.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return x, attns_list


class TransformerHead(nn.Module):
    def __init__(self, vit_params, c1_channels=256, c4_channels=2048, hid_dim=64, norm_layer=nn.BatchNorm2d):
        super().__init__()

        last_channels = vit_params['embed_dim']
        nhead = vit_params['num_heads']

        self.transformer = Transformer(vit_params, c4_channels=c4_channels)

        self.conv_c1 = _ConvBNReLU(c1_channels, hid_dim, 1, norm_layer=norm_layer)

        self.lay1 = SeparableConv2d(last_channels+nhead, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay2 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay3 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)

        self.pred = nn.Conv2d(hid_dim, 1, 1)


    def forward(self, x, c1):
        feat_enc, attns_list = self.transformer(x)
        attn_map = attns_list[-1]
        B, nclass, nhead, _ = attn_map.shape
        _, _, H, W = feat_enc.shape
        attn_map = attn_map.reshape(B*nclass, nhead, H, W)

        x = torch.cat([_expand(feat_enc, nclass), attn_map], 1)

        x = self.lay1(x)
        x = self.lay2(x)

        size = c1.size()[2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        c1 = self.conv_c1(c1)
        x = x + _expand(c1, nclass)

        x = self.lay3(x)
        x = self.pred(x).reshape(B, nclass, size[0], size[1])

        return x

def _expand(x, nclass):
    return x.unsqueeze(1).repeat(1, nclass, 1, 1, 1).flatten(0, 1)
