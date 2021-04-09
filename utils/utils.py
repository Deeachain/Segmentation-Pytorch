"""
Time:     2020/11/30 下午5:02
Author:   Ding Cheng(Deeachain)
File:     utils.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, paris_colorize_mask, road_colorize_mask, \
    austin_colorize_mask, isprs_colorize_mask


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):

    if output_grey:
        if dataset == 'cityscapes':
            output[np.where(output==18)] = 33
            output[np.where(output==17)] = 32
            output[np.where(output==16)] = 31
            output[np.where(output==15)] = 28
            output[np.where(output==14)] = 27
            output[np.where(output==13)] = 26
            output[np.where(output==12)] = 25
            output[np.where(output==11)] = 24
            output[np.where(output==10)] = 23
            output[np.where(output==9)] = 22
            output[np.where(output==8)] = 21
            output[np.where(output==7)] = 20
            output[np.where(output==6)] = 19
            output[np.where(output==5)] = 17
            output[np.where(output==4)] = 13
            output[np.where(output==3)] = 12
            output[np.where(output==2)] = 11
            output[np.where(output==1)] = 8
            output[np.where(output==0)] = 7
        output_grey = Image.fromarray(output)
        output_grey.save(os.path.join(save_path, img_name + '.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'paris':
            output_color = paris_colorize_mask(output)
        elif dataset == 'road':
            output_color = road_colorize_mask(output)
        elif dataset == 'austin':
            output_color = austin_colorize_mask(output)
        elif dataset == 'postdam' or dataset == 'vaihingen':
            output_color = isprs_colorize_mask(output)
        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'paris':
            gt_color = paris_colorize_mask(gt)
        elif dataset == 'road':
            gt_color = road_colorize_mask(gt)
        elif dataset == 'austin':
            gt_color = austin_colorize_mask(gt)
        elif dataset == 'postdam' or dataset == 'vaihingen':
            gt_color = isprs_colorize_mask(gt)
        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters
