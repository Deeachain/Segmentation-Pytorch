import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from tqdm import tqdm
from utils.convert_state import convert_state_dict


def predict(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    data_list = []
    pbar = tqdm(iterable=enumerate(test_loader), total=total_batches, desc='Predicting')
    for i, (input, label, size, name) in pbar:
        with torch.no_grad():
            input_var = input.cuda().float()
        output = model(input_var)
        torch.cuda.synchronize()

        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)

        # Save the predict greyscale output for Cityscapes official evaluation
        # Modify image name to meet official requirement
        save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                     output_grey=False, output_color=True, gt_color=False)
        data_list.append([gt.flatten(), output.flatten()])
    meanIoU, per_class_iu = get_iou(data_list, args.classes)
    print('miou {}\nclass iou {}'.format(meanIoU, per_class_iu))
    result = args.save_seg_dir + '/results.txt'
    with open(result, 'w') as f:
        f.write(str(meanIoU))
        f.write('\n{}'.format(str(per_class_iu)))


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print(">>>>>>>>>>beginning testing>>>>>>>>>>>")
    print("test set length: ", len(testLoader))
    predict(args, testLoader, model)


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', default="ENet", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="paris", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default="/media/ding/Study/graduate/code/Efficient-Segmentation-Networks/checkpoint/paris/ENetbs16gpu1_train/model_100.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./outputs/",
                        help="saving path of prediction result")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    save_dirname = args.checkpoint.split('/')[-2] + '_' + args.checkpoint.split('/')[-1].split('.')[0]
    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', save_dirname)


    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'paris':
        args.classes = 3
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)
