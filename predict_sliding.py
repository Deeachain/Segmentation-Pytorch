import numpy as np
import torch
import os
from math import ceil
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_sliding_test
from utils.utils import save_predict
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.metric.metric import get_iou


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]  # 512-512 = 0
    cols_missing = target_size[1] - img.shape[3]  # 512-512 = 0
    # 在右、下边用0padding
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img  # shape(1,3,512,512)


def pad_label(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[1]  # 512-512 = 0
    cols_missing = target_size[1] - img.shape[2]  # 512-512 = 0
    # 在右、下边用0padding
    padded_img = np.pad(img, ((0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img  # shape(1,512,512)


# 滑动窗口法
# image.shape(1,3,1024,2048)、tile_size=(512,512)、classes=3、flip=True、recur=1
# image:需要预测的图片(1,3,3072,3328);tile_size:小方块大小;
def predict_sliding(args, net, image, tile_size, classes):
    total_batches = len(image)
    data_list = []
    pbar = tqdm(iterable=enumerate(image), total=total_batches, desc='Predicting')
    for i, (input, gt, size, name) in pbar:
        image_size = input.shape  # (1,3,3328,3072)
        overlap = 1 / 3  # 每次滑动的覆盖率为1/3
        # print(image_size, tile_size)
        stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:512*(1-1/3) = 513     512*(1-1/3)= 342
        tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(3072-512)/342+1=9
        tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(3328-512)/342+1=10
        full_probs = np.zeros((image_size[2], image_size[3], classes))  # 初始化全概率矩阵shape(3072,3328,3)
        count_predictions = np.zeros((image_size[2], image_size[3], classes))  # 初始化计数矩阵shape(3072,3328,3)

        for row in range(tile_rows):  # row = 0,1     0,1,2,3,4,5,6,7,8
            for col in range(tile_cols):  # col = 0,1,2,3     0,1,2,3,4,5,6,7,8,9
                x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0   0*342
                y1 = int(row * stride)  # y1 = 0 * 513 = 0   0*342
                x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+512, 3328)
                y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+512, 3072)
                x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(512-512, 0)
                y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(512-512, 0)

                img = input[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:512, 0:512]
                padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为512*512
                # plt.imshow(padded_img)
                # plt.show()

                # 将扣下来的部分传入网络，网络输出概率图。
                with torch.no_grad():
                    input_var = torch.from_numpy(padded_img).cuda().float()
                    padded_prediction = net(input_var)

                    if type(padded_prediction) is tuple:
                        padded_prediction = padded_prediction[0]

                    torch.cuda.synchronize()

                if isinstance(padded_prediction, list):
                    padded_prediction = padded_prediction[0]  # shape(1,3,512,512)

                padded_prediction = padded_prediction.cpu().data[0].numpy().transpose(1, 2, 0)  # 通道位置变换(512,512,3)
                prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]  # 扣下相应面积 shape(512,512,3)
                count_predictions[y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
                full_probs[y1:y2, x1:x2] += prediction  # 窗口区域内的全概率矩阵叠加预测结果

        # average the predictions in the overlapping regions
        full_probs /= count_predictions  # 全概率矩阵 除以 计数矩阵 即得 平均概率
        # visualize normalization Weights
        # plt.imshow(np.mean(count_predictions, axis=2))
        # plt.show()
        full_probs = np.asarray(np.argmax(full_probs, axis=2), dtype=np.uint8)
        '''设置输出原图和预测图片的颜色灰度还是彩色'''
        gt = gt[0].numpy()
        # 计算miou
        data_list.append([gt.flatten(), full_probs.flatten()])
        save_predict(full_probs, gt, name[0], args.dataset, args.save_seg_dir,
                     output_grey=False, output_color=True, gt_color=True)

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
        # print(args.gpus)
        # torch.cuda.set_device(0)
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
    datas, testLoader = build_dataset_sliding_test(args.dataset, args.num_workers, none_gt=True)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print(">>>>>>>>>>>beginning testing>>>>>>>>>>>")
    predict_sliding(args, model.eval(), image=testLoader, tile_size=(args.tile_size, args.tile_size),
                    classes=args.classes)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ENet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="paris", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--num_loss', type=int, default=1, help="the number of loss")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--tile_size', type=int, default=1024,
                        help=" the tile_size is when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default='/media/ding/Study/graduate/code/Efficient-Segmentation-Networks/checkpoint/paris/ENetbs16gpu1_train/model_91.pth',
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./outputs/",
                        help="saving path of prediction result")
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

    test_model(args)
