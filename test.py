import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
import glob
from tqdm import tqdm
from utils.convert_state import convert_state_dict
from utils.metric.SegmentationMetric import SegmentationMetric


def test(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)

    metric = SegmentationMetric(numClass=args.classes)
    pbar = tqdm(iterable=enumerate(test_loader), total=total_batches, desc='Valing')
    for i, (input, gt, size, name) in pbar:
        with torch.no_grad():
            input_var = Variable(input).cuda()

        output = model(input_var)
        torch.cuda.synchronize()

        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        gt = np.asarray(gt[0], dtype=np.uint8)

        # 计算miou
        metric.addBatch(imgPredict=output.flatten(), imgLabel=gt.flatten())

        # save the predicted image
        if args.save:
            save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=True)

    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    Miou, PerMiou_set = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()

    return Miou, PerMiou_set, FWIoU, pa, mpa


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)
    mIOU_val_max = 0
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

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    if args.best:
        # Get the best test result among the all model records.
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                dirname, basename = os.path.split(args.checkpoint)
                mIOU_val = []
                check_num = []
                checkpoint_name = glob.glob(dirname + '/*.pth')
                for i in checkpoint_name:
                    name = i.split('/')[-1].split('_')[-1].split('.')[0]
                    check_num.append(int(name))
                check_num.sort()

                for i in check_num:
                    basename = 'model_' + str(i) + '.pth'
                    resume = os.path.join(dirname, basename)
                    checkpoint = torch.load(resume)
                    model.load_state_dict(checkpoint['model'])
                    print("beginning test the:" + basename)
                    print("validation set length: ", len(testLoader))
                    miou, class_iou, fmiou, pa, mpa = test(args, testLoader, model)
                    print('Miou Val is ', miou)
                    mIOU_val.append(miou)

                # index = list(range(epoch - 19, epoch + 1))[np.argmax(mIOU_val)]
                index = check_num[np.argmax(mIOU_val)]
                print("The best mIoU among the models is", index)
                mIOU_val_max = np.max(mIOU_val)

            else:
                print("no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # Save the result
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        args.logFile = 'test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a+')
    else:
        logger = open(logFileLoc, 'w')
    logger.write("Max Mean IoU: %.4f" % mIOU_val_max)

    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ENet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="paris", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=0,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default="/media/ding/Study/graduate/code/Efficient-Segmentation-Networks/checkpoint/paris/ENetbs8gpu1_train/model_300.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="outputs",
                        help="saving path of prediction result")
    parser.add_argument('--best', default=True, help="Get the best result among last few checkpoints")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    save_dirname = args.checkpoint.split('/')[-3] + '_' + args.checkpoint.split('/')[-1].split('.')[0]
    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', save_dirname)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'paris':
        args.classes = 3
    elif args.dataset == 'lake':
        args.classes = 2
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)
