from .resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from .xception import Xception, xception39


def build_backbone(backbone, out_stride=32, mult_grid=False):
    if backbone == 'resnet18':
        return resnet18(out_stride, mult_grid)
    elif backbone == 'resnet34':
        return resnet34(out_stride, mult_grid)
    elif backbone == 'resnet50':
        return resnet50(out_stride, mult_grid)
    elif backbone == 'resnet101':
        return resnet101(out_stride, mult_grid)
    else:
        raise NotImplementedError
