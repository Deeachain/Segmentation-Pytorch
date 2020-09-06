import os, sys
import time
import torch
from torch import optim
import torch.nn as nn
import timeit
import math
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight, netParams
from utils.metric.SegmentationMetric import SegmentationMetric
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR
from utils.earlyStopping import EarlyStopping
from tqdm import tqdm

sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
GLOBAL_SEED = 88


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """

    model.train()
    epoch_loss = []

    total_batches = len(train_loader)
    st = time.time()
    pbar = tqdm(iterable=enumerate(train_loader), total=total_batches,
                desc='Epoch {}/{}'.format(epoch, args.max_epochs))
    for iteration, batch in pbar:

        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration
        # learming scheduling
        if args.lr_schedule == 'poly':
            lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.lr_schedule == 'warmpoly':
            scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                     warmup_iters=args.warmup_iters, power=0.9)

        lr = optimizer.param_groups[0]['lr']

        images, labels, _, _ = batch

        images = images.cuda()
        labels = labels.long().cuda()
        if args.model == 'PSPNet50':
            x, aux = model(images)
            main_loss = criterion(x, labels)
            aux_loss = criterion(aux, labels)
            loss = 0.6 * main_loss + 0.4 * aux_loss
        else:
            output = model(images)
            if type(output) is tuple:
                output = output[0]
            loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item())

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def val(args, val_loader, criteria, model):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    val_loss = []
    metric = SegmentationMetric(args.classes)
    pbar = tqdm(iterable=enumerate(val_loader), total=total_batches, desc='Val')
    for iteration, (input, label, size, name) in pbar:
        with torch.no_grad():
            input_var = input.cuda().float()
            output = model(input_var)
            if type(output) is tuple:
                output = output[0]

        loss = criteria(output, label.long().cuda())
        val_loss.append(loss)
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        # 计算miou
        metric.addBatch(imgPredict=output.flatten(), imgLabel=gt.flatten())

    val_loss = sum(val_loss) / len(val_loss)

    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    Miou, PerMiou_set = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()

    return val_loss, FWIoU, Miou, PerMiou_set


def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("input size:{}".format(input_size))

    print(args)

    if args.cuda:
        print("use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # set the seed
    setup_seed(GLOBAL_SEED)
    print("set Global Seed: ", GLOBAL_SEED)
    cudnn.enabled = True
    print("building network")

    # build the model and initialization
    model = build_model(args.model, num_classes=args.classes)
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')

    print("computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train(args.dataset, input_size, args.batch_size, args.train_type,
                                                        args.random_scale, args.random_mirror, args.num_workers)

    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter

    print('Dataset statistics')
    print("data['classWeights']: ", datas['classWeights'])
    print('mean and std: ', datas['mean'], datas['std'])

    # define loss function, respectively
    weight = torch.from_numpy(datas['classWeights'])

    if args.dataset == 'camvid':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif args.dataset == 'camvid' and args.use_label_smoothing:
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)

    elif args.dataset == 'cityscapes' and args.use_ohem:
        min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
    elif args.dataset == 'cityscapes' and args.use_label_smoothing:
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
    elif args.dataset == 'cityscapes' and args.use_lovaszsoftmax:
        criteria = LovaszSoftmax(ignore_index=ignore_label)
    elif args.dataset == 'cityscapes' and args.use_focal:
        criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)

    elif args.dataset == 'paris':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)

    elif args.dataset == 'road':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif args.dataset == 'ai':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif args.dataset == 'ai' and args.use_ohem:
        min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(use_weight=True, weight=weight, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    if args.cuda:
        criteria = criteria.cuda()
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            args.gpu_nums = torch.cuda.device_count()
            model = nn.DataParallel(model).cuda()  # multi-card data parallel
        else:
            args.gpu_nums = 1
            print("single GPU for training")
            model = model.cuda()  # 1-card data parallel

    args.savedir = (args.savedir + args.dataset + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu_nums) + "_" + str(args.train_type) + '/')

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    with open(args.savedir + 'args.txt', 'w') as f:
        f.write('mean:{}\nstd:{}\n'.format(datas['mean'], datas['std']))
        f.write("Parameters: {} Seed: {}\n".format(str(total_paramters), GLOBAL_SEED))
        f.write(str(args))

    start_epoch = 0
    # continue training
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
            print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True # 寻找最优配置
    cudnn.deterministic = True # 减少波动

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=50)

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("%s\t%s\t\t%s\t%s\t%s\t%s\n" % ('Epoch', '   lr', 'Loss(Tr)', 'Loss(Val)', 'FWIOU(Val)', 'mIOU(Val)'))
    logger.flush()

    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'radam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.90, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'ranger':
        optimizer = Ranger(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.95, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)

    lossTr_list = []
    epoches = []
    mIOU_val_list = []
    lossVal_list = []
    print('>>>>>>>>>>>beginning training>>>>>>>>>>>')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        lossTr_list.append(lossTr)

        # validation
        if epoch % args.val_epochs == 0 or epoch == args.max_epochs-1:
            epoches.append(epoch)
            val_loss, FWIoU, mIOU_val, per_class_iu = val(args, valLoader, criteria, model)
            mIOU_val_list.append(mIOU_val)
            lossVal_list.append(val_loss.item())
            # record train information
            logger.write(
                "%d\t%.6f\t%.4f\t\t%.4f\t\t%0.4f\t\t%0.4f\t\t%s\n" % (epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, per_class_iu))
            logger.flush()
            print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\tVal Loss = %.4f\tFWIOU(val) = %.4f\tmIOU(val) = %.4f\tper_class_iu= %s\n" % (
                epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, str(per_class_iu)))
        else:
            # record train information
            logger.write("%d\t%.6f\t%.4f\n" % (epoch, lr, lossTr))
            logger.flush()
            print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\n" % (epoch, lr, lossTr))

        # save the model
        model_file_name = args.savedir + '/model_' + str(epoch) + '.pth'
        state = {"epoch": epoch, "model": model.state_dict()}

        # Individual Setting for save model
        if epoch >= args.max_epochs - 10:
            torch.save(state, model_file_name)
        elif epoch % 10 == 0:
            torch.save(state, model_file_name)

        # draw plots for visualization
        if os.path.isfile(args.savedir + "loss.png"):
            f = open(args.savedir + 'log.txt', 'r')
            next(f)
            epoch_list = []
            lossTr_list = []
            lossVal_list = []
            for line in f.readlines():
                epoch_list.append(float(line.strip().split()[0]))
                lossTr_list.append(float(line.strip().split()[2]))
                lossVal_list.append(float(line.strip().split()[3]))
            assert len(epoch_list) == len(lossTr_list) == len(lossVal_list)

            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(0, epoch + 1), lossTr_list, label='Train_loss')
            ax1.plot(range(0, epoch + 1), lossVal_list, label='Val_loss')
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")
            ax1.legend()

            plt.savefig(args.savedir + "loss.png")
            plt.close('all')
            plt.clf()
        else:
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(0, epoch + 1), lossTr_list, label='Train_loss')
            ax1.plot(range(0, epoch + 1), lossVal_list, label='Val_loss')
            ax1.set_title("Average loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")
            ax1.legend()

            plt.savefig(args.savedir + "loss.png")
            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(epoches, mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            ax2.legend()

            plt.savefig(args.savedir + "mIou.png")
            plt.close('all')

        early_stopping.monitor(monitor=mIOU_val)
        if early_stopping.early_stop:
            print("Early stopping and Save checkpoint")
            if not os.path.exists(model_file_name):
                torch.save(state, model_file_name)
                val_loss, mIOU_val, per_class_iu = val(args, valLoader, criteria, model, epoch)
                print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\tVal Loss = %.4f\tmIOU(val) = %.4f\tper_class_iu= %s\n" % (
                        epoch, lr, lossTr, val_loss, mIOU_val, str(per_class_iu)))
            break

    logger.close()


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default="ENet", help="model name: (default ENet)")
    parser.add_argument('--dataset', type=str, default="paris", help="dataset: cityscapes or camvid")
    parser.add_argument('--input_size', type=str, default="360,480", help="input size of model")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=3,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--train_type', type=str, default="train",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=300,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--val_epochs', type=int, default=1,
                        help="the number of epochs: 100 for val set")
    parser.add_argument('--random_mirror', type=bool, default=False, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=False, help="input image resize 0.5 to 2")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=8, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam', 'radam', 'ranger'],
                        help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='warmpoly', help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9, help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', default=False,
                        help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_ohem', default=False,
                        help='OhemCrossEntropy2d Loss for cityscapes dataset')
    parser.add_argument('--use_lovaszsoftmax', default=False,
                        help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', default=False, help=' FocalLoss2d for cityscapes dataset')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    # checkpoint and log
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_args()

    if args.dataset == 'cityscapes':
        args.classes = 19
        args.input_size = '512,1024'
        ignore_label = 255
    elif args.dataset == 'camvid':
        args.classes = 11
        args.input_size = '360,480'
        ignore_label = 11
    elif args.dataset == 'paris':
        args.classes = 3
        args.input_size = '512,512'
        ignore_label = 255
    elif args.dataset == 'road':
        args.classes = 2
        args.input_size = '512,512'
        ignore_label = 255
    elif args.dataset == 'ai':
        args.classes = 8
        args.input_size = '256,256'
        ignore_label = 255
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    train_model(args)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    print("training time: %d hour %d minutes" % (int(hour), int(minute)))
