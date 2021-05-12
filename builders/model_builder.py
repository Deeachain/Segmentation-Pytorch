# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     model_builder.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from model.UNet import UNet
from model.SegNet import SegNet
from model.FCN8s import FCN
from model.BiSeNet import BiSeNet
from model.BiSeNetV2 import BiSeNetV2
from model.PSPNet.pspnet import PSPNet
from model.DeeplabV3Plus import Deeplabv3plus_res50
from model.FCN_ResNet import FCN_ResNet
from model.DDRNet import DDRNet
from model.HRNet import HighResolutionNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def build_model(model_name, num_classes, backbone='resnet18', pretrained=False, out_stride=32, mult_grid=False):
    if model_name == 'FCN':
        model = FCN(num_classes=num_classes)
    elif model_name == 'FCN_ResNet':
        model = FCN_ResNet(num_classes=num_classes, backbone=backbone, out_stride=out_stride, mult_grid=mult_grid)
    elif model_name == 'SegNet':
        model = SegNet(classes=num_classes)
    elif model_name == 'UNet':
        model = UNet(num_classes=num_classes)
    elif model_name == 'BiSeNet':
        model = BiSeNet(num_classes=num_classes, backbone=backbone)
    elif model_name == 'BiSeNetV2':
        model = BiSeNetV2(num_classes=num_classes)
    elif model_name == 'HRNet':
        model = HighResolutionNet(num_classes=num_classes)
    elif model_name == 'Deeplabv3plus_res50':
        model = Deeplabv3plus_res50(num_classes=num_classes, os=out_stride, pretrained=True)
    elif model_name == "DDRNet":
        model = DDRNet(pretrained=True, num_classes=num_classes)
    elif model_name == 'PSPNet_res50':
        model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, num_classes=num_classes, zoom_factor=8, use_ppm=True,
                       pretrained=True)
    elif model_name == 'PSPNet_res101':
        model = PSPNet(layers=101, bins=(1, 2, 3, 6), dropout=0.1, num_classes=num_classes, zoom_factor=8, use_ppm=True,
                       pretrained=True)
    # elif model_name == 'PSANet50':
    #     return PSANet(layers=50, dropout=0.1, classes=num_classes, zoom_factor=8, use_psa=True, psa_type=2, compact=compact,
    #                shrink_factor=shrink_factor, mask_h=mask_h, mask_w=mask_w, psa_softmax=True, pretrained=True)

    if pretrained:
        checkpoint = model_zoo.load_url(model_urls[backbone])
        model_dict = model.state_dict()
        # print(model_dict)
        # Screen out layers that are not loaded
        pretrained_dict = {'backbone.' + k: v for k, v in checkpoint.items() if 'backbone.' + k in model_dict}
        # Update the structure dictionary for the current network
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


