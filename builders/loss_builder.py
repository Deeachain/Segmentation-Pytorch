import torch
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d


def build_loss(args, datas, ignore_label):
    if args.dataset == 'cityscapes':
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                                    0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                    1.0865, 1.1529, 1.0507])
    else:
        weight = torch.from_numpy(datas['classWeights'])
    # 默认使用交叉商损失函数
    criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    if args.use_ohem:
        min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(weight=weight, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
    elif args.use_label_smoothing:
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
    elif args.use_lovaszsoftmax:
        criteria = LovaszSoftmax(ignore_index=ignore_label)
    elif args.use_focal:
        criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)

    return criteria
