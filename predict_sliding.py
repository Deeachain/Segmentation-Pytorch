import os
import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from builders.loss_builder import build_loss
from builders.validation_builder import predict_whole, predict_sliding, predict_multiscale


def main(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        print("use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    testLoader, class_dict_df = build_dataset_test(args.dataset, args.num_workers, sliding=True, none_gt=True)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # define loss function, respectively
    criteria = build_loss(args, None, 255)

    print(">>>>>>>>>>>beginning testing>>>>>>>>>>>")
    val_loss, FWIoU, mIOU_val, per_class_iu = predict_sliding(args, model.eval(), testLoader=testLoader,
                                                              tile_size=(args.tile_size, args.tile_size),
                                                              criteria=criteria, mode='validation')

    t = PrettyTable(['label_index', 'class_name', 'class_iou'])
    for index in range(class_dict_df.shape[0]):
        t.add_row([class_dict_df['label_index'][index], class_dict_df['class_name'][index], PerMiou_set[index]])
    print(t.get_string(title="Miou is {}".format(mIOU_val)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="UNet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="paris", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--tile_size', type=int, default=512,
                        help=" the tile_size is when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default='/media/ding/Study/graduate/Segmentation_Torch/checkpoint/paris/UNetbs8gpu1_train/model_250.pth',
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./outputs/",
                        help="saving path of prediction result")
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss2d",
                        choices=['CrossEntropyLoss2d', 'ProbOhemCrossEntropy2d', 'CrossEntropyLoss2dLabelSmooth',
                                 'LovaszSoftmax', 'FocalLoss2d'], help="choice loss for train or val in list")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    save_dirname = args.checkpoint.split('/')[-2] + '_' + args.checkpoint.split('/')[-1].split('.')[0]
    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict_sliding', save_dirname)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'paris':
        args.classes = 3
    elif args.dataset == 'austin':
        args.classes = 2
    elif args.dataset == 'road':
        args.classes = 2
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    main(args)
