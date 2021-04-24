# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     predict.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from argparse import ArgumentParser
from prettytable import PrettyTable
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from builders.loss_builder import build_loss
from builders.validation_builder import predict_multiscale_sliding


def main(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    t = PrettyTable(['args_name', 'args_value'])
    for k in list(vars(args).keys()):
        t.add_row([k, vars(args)[k]])
    print(t.get_string(title="Predict Arguments"))

    # build the model
    model = build_model(args.model, args.classes, args.backbone, args.pretrained, args.out_stride, args.mult_grid)

    # load the test set
    if args.predict_type == 'validation':
        testdataset, class_dict_df = build_dataset_test(args.root, args.dataset, args.crop_size,
                                                        mode=args.predict_mode, gt=True)
    else:
        testdataset, class_dict_df = build_dataset_test(args.root, args.dataset, args.crop_size,
                                                        mode=args.predict_mode, gt=False)
    DataLoader = data.DataLoader(testdataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.batch_size, pin_memory=True, drop_last=False)

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        model = model.cuda()
        cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)['model']
            check_list = [i for i in checkpoint.items()]
            # Read weights with multiple cards, and continue training with a single card this time
            if 'module.' in check_list[0][0]:  # 读取使用多卡训练权重,并且此次使用单卡预测
                new_stat_dict = {}
                for k, v in checkpoint.items():
                    new_stat_dict[k[7:]] = v
                model.load_state_dict(new_stat_dict, strict=True)
            # Read the training weight of a single card, and continue training with a single card this time
            else:
                model.load_state_dict(checkpoint)
        else:
            print("no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # define loss function
    criterion = build_loss(args, None, 255)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
          ">>>>>>>>>>>  beginning testing   >>>>>>>>>>>>\n"
          ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    predict_multiscale_sliding(args=args, model=model, testLoader=DataLoader, class_dict_df=class_dict_df,
                                scales=args.scales, overlap=args.overlap, criterion=criterion,
                                mode=args.predict_type, save_result=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="UNet", help="model name")
    parser.add_argument('--backbone', type=str, default="resnet18", help="backbone name")
    parser.add_argument('--pretrained', action='store_true',
                        help="whether choice backbone pretrained on imagenet")
    parser.add_argument('--out_stride', type=int, default=32, help="output stride of backbone")
    parser.add_argument('--mult_grid', action='store_true',
                        help="whether choice mult_grid in backbone last layer")
    parser.add_argument('--root', type=str, default="", help="path of datasets")
    parser.add_argument('--predict_mode', default="sliding", choices=["sliding", "whole"],
                        help="Defalut use whole predict mode")
    parser.add_argument('--predict_type', default="validation", choices=["validation", "predict"],
                        help="Defalut use validation type")
    parser.add_argument('--flip_merge', action='store_true', help="Defalut use predict without flip_merge")
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0], help="predict with multi_scales")
    parser.add_argument('--overlap', type=float, default=0.0, help="sliding predict overlap rate")
    parser.add_argument('--dataset', default="paris", help="dataset: cityscapes")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing NOTES:image size should fixed!")
    parser.add_argument('--tile_hw_size', type=str, default='512, 512',
                        help=" the tile_size is when evaluating or testing")
    parser.add_argument('--crop_size', type=int, default=769, help="crop size of image")
    parser.add_argument('--input_size', type=str, default=(769, 769),
                        help=" the input_size is for build ProbOhemCrossEntropy2d loss")
    parser.add_argument('--checkpoint', type=str, default='',
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./outputs/",
                        help="saving path of prediction result")
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss2d",
                        choices=['CrossEntropyLoss2d', 'ProbOhemCrossEntropy2d', 'CrossEntropyLoss2dLabelSmooth',
                                 'LovaszSoftmax', 'FocalLoss2d'], help="choice loss for train or val in list")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    save_dirname = args.checkpoint.split('/')[-2] + '_' + args.checkpoint.split('/')[-1].split('.')[0]

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.predict_mode, save_dirname)

    if args.dataset == 'cityscapes':
        args.classes = 19
    else:
        raise NotImplementedError(
            "This repository now supports datasets %s is not included" % args.dataset)

    main(args)
