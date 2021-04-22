# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 下午5:02
Author:   Ding Cheng(Deeachain)
File:     validation_builder.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import outer
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
from prettytable import PrettyTable
import torch.nn.functional as F
import cv2
from math import ceil
from tqdm import tqdm
from utils.utils import save_predict
from utils.metric.SegmentationMetric import SegmentationMetric


def eval_metric(args, class_dict_df, metric, count_loss, loss):
    loss /= count_loss
    Pa = metric.pixelAccuracy()
    Mpa, Cpa = metric.meanPixelAccuracy()
    Miou, Ciou = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    Precision = metric.precision()
    Recall = metric.recall()
    F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    MF = np.nanmean(F1)

    Pa = np.around(Pa, decimals=4)
    Miou = np.around(Miou, decimals=4)
    Precision = np.around(Precision, decimals=4)
    Recall = np.around(Recall, decimals=4)
    P = np.sum(Precision[:]) / len(Precision[:])
    R = np.sum(Recall[:]) / len(Recall[:])
    F = np.sum(F1[:]) / len(F1[:])

    PerCiou_set = {}
    PerCiou = np.around(Ciou, decimals=4)
    for index, per in enumerate(PerCiou):
        PerCiou_set[index] = per

    PerCpa_set = {}
    PerCpa = np.around(Cpa, decimals=4)
    for index, per in enumerate(PerCpa):
        PerCpa_set[index] = per

    F_set = {}
    F1 = np.around(F1, decimals=4)
    for index, per in enumerate(F1):
        F_set[index] = per

    if args.dataset == "postdam" or args.dataset == 'vaihingen':
        Miou_Noback = np.sum(Ciou[:5]) / len(Ciou[:5])
        P_Noback = np.sum(Precision[:5]) / len(Precision[:5])
        R_Noback = np.sum(Recall[:5]) / len(Recall[:5])
        F1_Noback = np.sum(F1[:5]) / len(F1[:5])
    elif args.dataset == 'paris':
        Miou_Noback = np.sum(Ciou[:]) / len(Ciou[:])
        P_Noback = np.sum(Precision[:]) / len(Precision[:])
        R_Noback = np.sum(Recall[:]) / len(Recall[:])
        F1_Noback = np.sum(F1[:]) / len(F1[:])
    elif args.dataset == 'cityscapes':
        Miou_Noback = np.sum(Ciou[:]) / len(Ciou[:])
        P_Noback = np.sum(Precision[:]) / len(Precision[:])
        R_Noback = np.sum(Recall[:]) / len(Recall[:])
        F1_Noback = np.sum(F1[:]) / len(F1[:])


    t = PrettyTable(['label_index', 'label_name', 'IoU', 'Precision', 'Recall', 'F1'])
    for key in PerCiou_set:
        t.add_row([key, class_dict_df['class_name'][key], PerCiou_set[key], Precision[key], Recall[key], F1[key]])
    print(t.get_string(title="Validation results"))
    print('OA:{:.4f}'
          '\nMIoU:{:.4f}       MIoU_Noback:{:.4f}'
          '\nPrecision:{:.4f}  Precision_Noback:{:.4f}'
          '\nRecall:{:.4f}     Recall_Noback:{:.4f}'
          '\nF1:{:.4f}         F1_Noback:{:.4f}'
          .format(Pa, Miou, Miou_Noback, P, P_Noback, R, R_Noback, F, F1_Noback))
          

    result = args.save_seg_dir + '/results.txt'
    with open(result, 'w') as f:
        f.write(str(t.get_string(title="Validation results")))
        f.write('\nOA:{:.4f}'
                '\nMIoU:{:.4f}       MIoU_Noback:{:.4f}'
                '\nPrecision:{:.4f}  Precision_Noback:{:.4f}'
                '\nRecall:{:.4f}     Recall_Noback:{:.4f}'
                '\nF1:{:.4f}         F1_Noback:{:.4f}\n'
                .format(Pa, Miou, Miou_Noback, P, P_Noback, R, R_Noback, F, F1_Noback))

    return loss, FWIoU, Miou, Miou_Noback, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_Noback


def aux_output(output):
    if type(output) is tuple:
        output = output[0]
    return output


def flip_merge(model, scale_image):
    output1 = model(scale_image)
    output1 = aux_output(output1)
    # 水平翻转
    output2 = model(torch.flip(scale_image, [-1]))
    output2 = aux_output(output2)
    output2 = torch.flip(output2, [-1])
    # 垂直翻转
    output3 = model(torch.flip(scale_image, [-2]))
    output3 = aux_output(output3)
    output3 = torch.flip(output3, [-2])
    # 水平垂直翻转
    output4 = model(torch.flip(scale_image, [-1, -2]))
    output4 = aux_output(output4)
    output4 = torch.flip(output4, [-1, -2])
    output = output1 + output2 + output3 + output4
    return output


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


def predict_multiscale_sliding(args, model, class_dict_df, testLoader, scales, overlap, criterion, mode='predict',
                               save_result=True):
    loss = 0
    count_loss = 0
    model.eval()
    criterion = criterion.cuda()
    tile_h_size, tile_w_size = int(args.tile_hw_size.split(',')[0]), int(args.tile_hw_size.split(',')[1])
    metric = SegmentationMetric(args.classes)  # args.classes表示有args.classes个分类
    pbar = tqdm(iterable=enumerate(testLoader), total=len(testLoader), desc='Predicting')
    if mode == 'validation':
        for i, (image, gt, size, name) in pbar:
            B, C, H, W = image.shape
            # image and gt scaled together [0.75, 1.0, 1.25, 1.5, 2.0]
            full_prob = torch.zeros(B, args.classes, H, W).cuda()
            for scale in scales:
                scale = float(scale)
                scale = float(scale)
                sh = int(H * scale)
                sw = int(W * scale)
                
                scale_image = F.interpolate(image, (sh, sw), mode='bilinear', align_corners=True).float()
                scale_gt = F.interpolate(gt.unsqueeze(1).float(), (sh, sw), mode='nearest').long()

                # scale之后的尺寸是否大于title_size
                if (H > sh or W > sw) and (H < tile_h_size or W < tile_w_size):
                    # 直接整张图片预测，并且还原为正常尺寸
                    with torch.no_grad():
                        scale_image = scale_image.cuda()
                        scale_gt = scale_gt.cuda()
                        if args.flip_merge:
                            outputs = flip_merge(model, scale_image)
                        else:
                            outputs = model(scale_image)
                        if type(outputs) is tuple:
                            length = len(outputs)
                            for index, out in enumerate(outputs):
                                criterion = criterion.cuda()
                                loss_record = criterion(out, scale_gt.squeeze(1))
                                if index == 0:
                                    loss_record *= 0.6
                                else:
                                    loss_record *= 0.4 / (length - 1)
                                loss += loss_record
                            outputs = outputs[0]
                            count_loss += 1
                        elif type(outputs) is not tuple:
                            loss += criterion(outputs, scale_gt.squeeze(1))
                            count_loss += 1
                else:
                    scale_image_size = scale_image.shape  # (b,c,h,w)
                    # overlap表示每次滑动的覆盖率
                    stride = ceil(tile_h_size * (1 - overlap))  # 滑动步长:512*(1-1/3) = 513     512*(1-1/3)= 342
                    tile_rows = int(ceil((scale_image_size[2] - tile_h_size) / stride) + 1)  # 行滑动步数:(3072-512)/342+1=9
                    tile_cols = int(ceil((scale_image_size[3] - tile_w_size) / stride) + 1)  # 列滑动步数:(3328-512)/342+1=10
                    outputs_prob = torch.zeros(B, args.classes, sh, sw).cuda()
                    count_prob = torch.zeros(B, 1, sh, sw).cuda()
            
                    for row in range(tile_rows):  # row = 0,1     0,1,2,3,4,5,6,7,8
                        for col in range(tile_cols):  # col = 0,1,2,3     0,1,2,3,4,5,6,7,8,9
                            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0   0*342
                            y1 = int(row * stride)  # y1 = 0 * 513 = 0   0*342
                            x2 = min(x1 + tile_w_size, scale_image_size[3])  # 末位置x2 = min(0+512, 3328)
                            y2 = min(y1 + tile_h_size, scale_image_size[2])  # y2 = min(0+512, 3072)
                            x1 = max(int(x2 - tile_w_size), 0)  # 重新校准起始位置x1 = max(512-512, 0)
                            y1 = max(int(y2 - tile_h_size), 0)  # y1 = max(512-512, 0)

                            with torch.no_grad():
                                tile_image = scale_image[:, :, y1:y2, x1:x2].float().cuda()
                                tile_gt = scale_gt[:, :, y1:y2, x1:x2].long().cuda()
                                if args.flip_merge:
                                    tile_output = flip_merge(model, tile_image)
                                else:
                                    tile_output = model(tile_image)

                                    # output = (main_loss, aux_loss1, axu_loss2***)
                            if type(tile_output) is tuple:
                                length = len(tile_output)
                                for index, out in enumerate(tile_output):
                                    criterion = criterion.cuda()
                                    loss_record = criterion(out, tile_gt.squeeze(1))
                                    if index == 0:
                                        loss_record *= 0.6
                                    else:
                                        loss_record *= 0.4 / (length - 1)
                                    loss += loss_record
                                tile_output = tile_output[0]
                                count_loss += 1
                            elif type(tile_output) is not tuple:
                                loss += criterion(tile_output, tile_gt.squeeze(1))
                                count_loss += 1

                            outputs_prob[:, :, y1:y2, x1:x2] += tile_output
                            count_prob[:, :, y1:y2, x1:x2] += 1

                    # 结束每一个scale之后的图片滑动窗口计算概率
                    assert ((count_prob == 0).sum() == 0)
                    outputs = outputs_prob / count_prob

                outputs = F.interpolate(outputs, (H, W), mode='bilinear', align_corners=True)
                full_prob += outputs

            # visualize normalization Weights
            # plt.imshow(np.mean(count_predictions, axis=2))
            # plt.show()
            gt = np.asarray(gt.cpu(), dtype=np.uint8)
            full_prob = torch.argmax(full_prob, 1).long()
            full_prob = np.asarray(full_prob.cpu(), dtype=np.uint8)  # (B,C,H,W)

            # plt.imshow(gt[0])
            # plt.show()
            # 计算miou

            '''设置输出原图和预测图片的颜色灰度还是彩色'''
            for index in range(full_prob.shape[0]):  # full_prob shape[0] is batch_size
                metric.addBatch(full_prob[index], gt[index])
                if save_result:
                    save_predict(full_prob[index], gt[index], name[index], args.dataset, args.save_seg_dir,
                                 output_grey=False, output_color=True, gt_color=True)

        loss, FWIoU, Miou, Miou_Noback, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_Noback = \
            eval_metric(args, class_dict_df, metric, count_loss, loss)
    else:
        for i, (image, size, name) in pbar:
            B, C, H, W = image.shape
            # image scaled [0.75, 1.0, 1.25, 1.5, 2.0]
            full_prob = torch.zeros(B, args.classes, H, W).cuda()
            for scale in scales:
                scale = float(scale)
                sh = int(H * scale)
                sw = int(W * scale)

                scale_image = F.interpolate(image, (sh, sw), mode='bilinear', align_corners=True).float()

                # scale之后的尺寸是否大于title_size
                if (H > sh or W > sw) and (H < tile_h_size or W < tile_w_size):
                    # 直接整张图片预测，并且还原为正常尺寸
                    with torch.no_grad():
                        scale_image = scale_image.cuda()
                        if args.flip_merge:
                            outputs = flip_merge(model, scale_image)
                        else:
                            outputs = model(scale_image)
                        if type(outputs) is tuple:
                            outputs = outputs[0]

                else:
                    scale_image_size = scale_image.shape  # (b,c,h,w)
                    # overlap表示每次滑动的覆盖率
                    stride = ceil(tile_h_size * (1 - overlap))  # 滑动步长:512*(1-1/3) = 513     512*(1-1/3)= 342
                    tile_rows = int(ceil((scale_image_size[2] - tile_h_size) / stride) + 1)  # 行滑动步数:(3072-512)/342+1=9
                    tile_cols = int(ceil((scale_image_size[3] - tile_w_size) / stride) + 1)  # 列滑动步数:(3328-512)/342+1=10
                    outputs_prob = torch.zeros(B, args.classes, sh, sw).cuda()
                    count_prob = torch.zeros(B, 1, sh, sw).cuda()
            
                    for row in range(tile_rows):  # row = 0,1     0,1,2,3,4,5,6,7,8
                        for col in range(tile_cols):  # col = 0,1,2,3     0,1,2,3,4,5,6,7,8,9
                            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0   0*342
                            y1 = int(row * stride)  # y1 = 0 * 513 = 0   0*342
                            x2 = min(x1 + tile_w_size, scale_image_size[3])  # 末位置x2 = min(0+512, 3328)
                            y2 = min(y1 + tile_h_size, scale_image_size[2])  # y2 = min(0+512, 3072)
                            x1 = max(int(x2 - tile_w_size), 0)  # 重新校准起始位置x1 = max(512-512, 0)
                            y1 = max(int(y2 - tile_h_size), 0)  # y1 = max(512-512, 0)

                            with torch.no_grad():
                                tile_image = scale_image[:, :, y1:y2, x1:x2].float().cuda()
                                if args.flip_merge:
                                    tile_output = flip_merge(model, tile_image)
                                else:
                                    tile_output = model(tile_image)

                            if type(tile_output) is tuple:
                                tile_output = tile_output[0]

                            outputs_prob[:, :, y1:y2, x1:x2] += tile_output
                            count_prob[:, :, y1:y2, x1:x2] += 1

                    # 结束每一个scale之后的图片滑动窗口计算概率
                    assert ((count_prob == 0).sum() == 0)
                    outputs = outputs_prob / count_prob

                outputs = F.interpolate(outputs, (H, W), mode='bilinear', align_corners=True)
                full_prob += outputs

            # visualize normalization Weights
            # plt.imshow(np.mean(count_predictions, axis=2))
            # plt.show()
            full_prob = torch.argmax(full_prob, 1).long()
            full_prob = np.asarray(full_prob.cpu(), dtype=np.uint8)  # (B,C,H,W)

            # plt.imshow(gt[0])
            # plt.show()
            # 计算miou

            '''设置输出原图和预测图片的颜色灰度还是彩色'''
            # save results
            for index in range(full_prob.shape[0]):  # gt shape[0] is batch_size
                if save_result:
                    save_predict(full_prob[index], None, name[index], args.dataset, args.save_seg_dir,
                                 output_grey=True, output_color=False, gt_color=False)

        loss, FWIoU, Miou, Miou_Noback, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_Noback = 0, 0, 0, 0, {}, 0, {}, 0, 0, {}, 0

    return loss, FWIoU, Miou, Miou_Noback, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_Noback


def predict_overlap_sliding(args, model, testLoader, scales, criterion, mode='predict', save_result=True):
    loss = 0
    count_loss = 0
    model.eval()
    criterion = criterion.cuda()
    tile_h_size, tile_w_size = int(args.tile_hw_size.split(',')[0]), int(args.tile_hw_size.split(',')[1])
    center_h_size, center_w_size = int(args.center_hw_size.split(',')[0]), int(args.center_hw_size.split(',')[1])
    metric = SegmentationMetric(args.classes)  # args.classes表示有args.classes个分类
    pbar = tqdm(iterable=enumerate(testLoader), total=len(testLoader), desc='Predicting')
    if mode == 'validation':
        for i, (image, gt, size, name) in pbar:
            B, C, H, W = image.shape
            # image and gt scaled together [0.75, 1.0, 1.25, 1.5, 2.0]
            full_prob = torch.zeros(B, args.classes, H, W).cuda()
            for scale in scales:
                scale = float(scale)
                scale = float(scale)
                sh = int(H * scale)
                sw = int(W * scale)

                scale_image = F.interpolate(image, (sh, sw), mode='bilinear', align_corners=True)
                scale_gt = F.interpolate(gt.unsqueeze(1).float(), (sh, sw), mode='nearest').long()

                # scale之后的尺寸是否大于title_size
                if H < sh and W < sw and (H < tile_h_size or W < tile_w_size):
                    # 直接整张图片预测，并且还原为正常尺寸
                    scale_image = scale_image.cuda()
                    scale_gt = scale_gt.cuda()
                    if args.flip_merge:
                        scale_output = flip_merge(model, scale_image)
                    else:
                        scale_output = model(scale_image)
                else:
                    # 根据保留中心尺寸，检查图片是否需要padding，确保图片是中心尺寸的倍数，倍数*中心尺寸-512=padding*2
                    scale_h, scale_w = scale_image.shape[2], scale_image.shape[3]
                    if scale_h % center_h_size == 0 and scale_w % center_w_size == 0:
                        tile_rows = scale_h / center_h_size
                        tile_cols = scale_w / center_w_size
                    else:
                        h_times = scale_h // center_h_size + 1
                        w_times = scale_w // center_w_size + 1
                        scale_image = pad_image(scale_image, (h_times * center_h_size, w_times * center_w_size))
                        pad_scale_h, pad_scale_w = scale_image.shape[2], scale_image.shape[3]
                        tile_rows = pad_scale_h / center_h_size
                        tile_cols = pad_scale_w / center_w_size
                    # （输入尺寸-保留中心尺寸）// 2 == 大图padding
                    outer_h_padding = int((tile_h_size - center_h_size) / 2)
                    outer_w_padding = int((tile_w_size - center_w_size) / 2)
                    scale_image = pad_image(scale_image, (outer_h_padding, outer_w_padding))

                    scale_image_size = scale_image.shape  # (b,c,h,w)
                    overlap = 1 / 3  # 每次滑动的覆盖率为1/3
                    stride = ceil(tile_h_size * (1 - overlap))  # 滑动步长:512*(1-1/3) = 513     512*(1-1/3)= 342
                    tile_rows = int(ceil((scale_image_size[2] - tile_h_size) / stride) + 1)  # 行滑动步数:(3072-512)/342+1=9
                    tile_cols = int(ceil((scale_image_size[3] - tile_w_size) / stride) + 1)  # 列滑动步数:(3328-512)/342+1=10
                    outputs_prob = torch.zeros(B, args.classes, sh, sw).cuda()
                    count_prob = torch.zeros(B, 1, sh, sw).cuda()

                    for row in range(tile_rows):  # row = 0,1     0,1,2,3,4,5,6,7,8
                        for col in range(tile_cols):  # col = 0,1,2,3     0,1,2,3,4,5,6,7,8,9
                            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0   0*342
                            y1 = int(row * stride)  # y1 = 0 * 513 = 0   0*342
                            x2 = min(x1 + tile_w_size, scale_image_size[3])  # 末位置x2 = min(0+512, 3328)
                            y2 = min(y1 + tile_h_size, scale_image_size[2])  # y2 = min(0+512, 3072)
                            x1 = max(int(x2 - tile_w_size), 0)  # 重新校准起始位置x1 = max(512-512, 0)
                            y1 = max(int(y2 - tile_h_size), 0)  # y1 = max(512-512, 0)

                            with torch.no_grad():
                                tile_image = scale_image[:, :, y1:y2, x1:x2].cuda()
                                tile_gt = scale_gt[:, :, y1:y2, x1:x2].long().cuda()
                                if args.flip_merge:
                                    tile_output = flip_merge(model, tile_image)
                                else:
                                    tile_output = model(tile_image)

                                    # output = (main_loss, aux_loss1, axu_loss2***)
                            if type(tile_output) is tuple:
                                length = len(scale_output)
                                for index, scale_out in enumerate(scale_output):
                                    criterion = criterion.cuda()
                                    loss_record = criterion(scale_out, scale_gt.squeeze(1))
                                    if index == 0:
                                        loss_record *= 0.6
                                    else:
                                        loss_record *= 0.4 / (length - 1)
                                    loss += loss_record
                                scale_output = scale_output[0]
                                count_loss += 1
                            elif type(tile_output) is not tuple:
                                loss += criterion(tile_output, tile_gt.squeeze(1))
                                count_loss += 1

                            outputs_prob[:, :, y1:y2, x1:x2] += tile_output
                            count_prob[:, :, y1:y2, x1:x2] += 1

                    # 结束每一个scale之后的图片滑动窗口计算概率
                    assert ((count_prob == 0).sum() == 0)
                    outputs = outputs_prob / count_prob

                outputs = F.interpolate(outputs, (H, W), mode='bilinear', align_corners=True)
                full_prob += outputs

            # visualize normalization Weights
            # plt.imshow(np.mean(count_predictions, axis=2))
            # plt.show()
            gt = gt.cpu().numpy()
            full_prob = torch.argmax(full_prob, 1).long()
            full_prob = np.asarray(full_prob.cpu(), dtype=np.uint8)  # (B,C,H,W)

            # plt.imshow(gt[0])
            # plt.show()
            # 计算miou

            '''设置输出原图和预测图片的颜色灰度还是彩色'''
            for index in range(full_prob.shape[0]):  # full_prob shape[0] is batch_size
                metric.addBatch(full_prob[index], gt[index])
                if save_result:
                    save_predict(full_prob[index], gt[index], name[index], args.dataset, args.save_seg_dir,
                                 output_grey=False, output_color=True, gt_color=True)

        loss, FWIoU, Miou, MIoU_avg, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F_avg = eval_metric(args, metric,
                                                                                                      count_loss, loss)
    else:
        for i, (image, size, name) in pbar:
            B, C, H, W = image.shape
            # image scaled [0.75, 1.0, 1.25, 1.5, 2.0]
            full_prob = torch.zeros(B, args.classes, H, W).cuda()
            for scale in scales:
                sh = int(H * float(scale))
                sw = int(W * float(scale))
                scale_image = F.interpolate(image, (sh, sw), mode='bilinear', align_corners=True)

                # scale之后的尺寸是否大于title_size
                if H < sh and W < sw and (H < tile_h_size or W < tile_w_size):
                    # 直接整张图片预测，并且还原为正常尺寸
                    scale_image = scale_image.cuda()
                    scale_gt = scale_gt.cuda()
                    if args.flip_merge:
                        scale_output = flip_merge(model, scale_image)
                    else:
                        scale_output = model(scale_image)
                else:
                    scale_image_size = scale_image.shape  # (b,c,h,w)
                    overlap = 1 / 3  # 每次滑动的覆盖率为1/3
                    stride = ceil(tile_h_size * (1 - overlap))  # 滑动步长:512*(1-1/3)= 342
                    tile_rows = int(ceil((scale_image_size[2] - tile_h_size) / stride) + 1)  # 行滑动步数:(3072-512)/342+1=9
                    tile_cols = int(ceil((scale_image_size[3] - tile_w_size) / stride) + 1)  # 列滑动步数:(3328-512)/342+1=10
                    outputs_prob = torch.zeros(B, args.classes, sh, sw).cuda()
                    count_prob = torch.zeros(B, 1, sh, sw).cuda()

                    for row in range(tile_rows):  # row = 0,1     0,1,2,3,4,5,6,7,8
                        for col in range(tile_cols):  # col = 0,1,2,3     0,1,2,3,4,5,6,7,8,9
                            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0   0*342
                            y1 = int(row * stride)  # y1 = 0 * 513 = 0   0*342
                            x2 = min(x1 + tile_w_size, scale_image_size[3])  # 末位置x2 = min(0+512, 3328)
                            y2 = min(y1 + tile_h_size, scale_image_size[2])  # y2 = min(0+512, 3072)
                            x1 = max(int(x2 - tile_w_size), 0)  # 重新校准起始位置x1 = max(512-512, 0)
                            y1 = max(int(y2 - tile_h_size), 0)  # y1 = max(512-512, 0)

                            with torch.no_grad():
                                tile_image = scale_image[:, :, y1:y2, x1:x2].cuda()
                                tile_gt = scale_gt[:, :, y1:y2, x1:x2].long().cuda()
                                if args.flip_merge:
                                    tile_output = flip_merge(model, tile_image)
                                else:
                                    tile_output = model(tile_image)

                            if type(tile_output) is tuple:
                                tile_output = tile_output[0]

                            outputs_prob[:, :, y1:y2, x1:x2] += tile_output
                            count_prob[:, :, y1:y2, x1:x2] += 1

                    # 结束每一个scale之后的图片滑动窗口计算概率
                    assert ((count_prob == 0).sum() == 0)
                    outputs = outputs_prob / count_prob

                outputs = F.interpolate(outputs, (H, W), mode='bilinear', align_corners=True)
                full_prob += outputs

            # visualize normalization Weights
            # plt.imshow(np.mean(count_predictions, axis=2))
            # plt.show()
            gt = gt.cpu().numpy()
            full_prob = torch.argmax(full_prob, 1).long()
            full_prob = np.asarray(full_prob.cpu(), dtype=np.uint8)  # (B,C,H,W)

            # plt.imshow(gt[0])
            # plt.show()
            # 计算miou
            for index in range(full_prob.shape[0]):  # gt shape[0] is batch_size
                if save_result:
                    save_predict(full_prob[index], None, name[index], args.dataset, args.save_seg_dir,
                                 output_grey=True, output_color=False, gt_color=False)

        loss, FWIoU, Miou, MIoU_avg, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F_avg = 0, 0, 0, 0, {}, 0, {}, 0, 0, {}, 0

    return loss, FWIoU, Miou, MIoU_avg, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F_avg
