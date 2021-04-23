# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     loss_builder.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, LabelSmoothing


def build_loss(args, datas, ignore_label):
    if args.dataset == 'cityscapes':
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                                    0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                    1.0865, 1.1529, 1.0507])
    elif datas != None:
        weight = torch.from_numpy(datas['classWeights'])
    else:
        weight = None

    # Default uses cross quotient loss function
    criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    if args.loss == 'ProbOhemCrossEntropy2d':
        h, w = args.base_size, args.base_size
        min_kept = int(args.batch_size // len(args.gpus_id) * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(weight=weight, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
    elif args.loss == 'CrossEntropyLoss2dLabelSmooth':
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
        # criteria = LabelSmoothing()
    elif args.loss == 'LovaszSoftmax':
        criteria = LovaszSoftmax(ignore_index=ignore_label)
    elif args.loss == 'FocalLoss2d':
        criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)

    return criteria
