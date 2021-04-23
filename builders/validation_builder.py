# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
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

    output2 = model(torch.flip(scale_image, [-1]))
    output2 = aux_output(output2)
    output2 = torch.flip(output2, [-1])

    output3 = model(torch.flip(scale_image, [-2]))
    output3 = aux_output(output3)
    output3 = torch.flip(output3, [-2])

    output4 = model(torch.flip(scale_image, [-1, -2]))
    output4 = aux_output(output4)
    output4 = torch.flip(output4, [-1, -2])
    output = output1 + output2 + output3 + output4
    return output


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def pad_label(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[1]
    cols_missing = target_size[1] - img.shape[2]
    padded_img = np.pad(img, ((0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_multiscale_sliding(args, model, class_dict_df, testLoader, scales, overlap, criterion, mode='predict',
                               save_result=True):
    loss = 0
    count_loss = 0
    model.eval()
    criterion = criterion.cuda()
    tile_h_size, tile_w_size = int(args.tile_hw_size.split(',')[0]), int(args.tile_hw_size.split(',')[1])
    metric = SegmentationMetric(args.classes)
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

                # Whether the size after scale is greater than title_size
                if (H > sh or W > sw) and (H < tile_h_size or W < tile_w_size):
                    # Directly predict the entire image and restore it to normal size
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
                    # overlap stands for coverage per slide
                    stride = ceil(tile_h_size * (1 - overlap))
                    tile_rows = int(ceil((scale_image_size[2] - tile_h_size) / stride) + 1)
                    tile_cols = int(ceil((scale_image_size[3] - tile_w_size) / stride) + 1)
                    outputs_prob = torch.zeros(B, args.classes, sh, sw).cuda()
                    count_prob = torch.zeros(B, 1, sh, sw).cuda()
            
                    for row in range(tile_rows):
                        for col in range(tile_cols):
                            x1 = int(col * stride)
                            y1 = int(row * stride)
                            x2 = min(x1 + tile_w_size, scale_image_size[3])
                            y2 = min(y1 + tile_h_size, scale_image_size[2])
                            x1 = max(int(x2 - tile_w_size), 0)
                            y1 = max(int(y2 - tile_h_size), 0)

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

                    assert ((count_prob == 0).sum() == 0)
                    outputs = outputs_prob / count_prob

                outputs = F.interpolate(outputs, (H, W), mode='bilinear', align_corners=True)
                full_prob += outputs

            gt = np.asarray(gt.cpu(), dtype=np.uint8)
            full_prob = torch.argmax(full_prob, 1).long()
            full_prob = np.asarray(full_prob.cpu(), dtype=np.uint8)  # (B,C,H,W)

            '''Sets the color of the output image and predict image to be grayscale or color'''
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

                # Whether the size after scale is greater than title_size
                if (H > sh or W > sw) and (H < tile_h_size or W < tile_w_size):
                    # Directly predict the entire image and restore it to normal size
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
                    # overlap stands for coverage per slide
                    stride = ceil(tile_h_size * (1 - overlap))
                    tile_rows = int(ceil((scale_image_size[2] - tile_h_size) / stride) + 1)
                    tile_cols = int(ceil((scale_image_size[3] - tile_w_size) / stride) + 1)
                    outputs_prob = torch.zeros(B, args.classes, sh, sw).cuda()
                    count_prob = torch.zeros(B, 1, sh, sw).cuda()
            
                    for row in range(tile_rows):
                        for col in range(tile_cols):
                            x1 = int(col * stride)
                            y1 = int(row * stride)
                            x2 = min(x1 + tile_w_size, scale_image_size[3])
                            y2 = min(y1 + tile_h_size, scale_image_size[2])
                            x1 = max(int(x2 - tile_w_size), 0)
                            y1 = max(int(y2 - tile_h_size), 0)

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

                    assert ((count_prob == 0).sum() == 0)
                    outputs = outputs_prob / count_prob

                outputs = F.interpolate(outputs, (H, W), mode='bilinear', align_corners=True)
                full_prob += outputs

            full_prob = torch.argmax(full_prob, 1).long()
            full_prob = np.asarray(full_prob.cpu(), dtype=np.uint8)  # (B,C,H,W)

            '''Sets the color of the output image and predict image to be grayscale or color'''
            # save results
            for index in range(full_prob.shape[0]):  # gt shape[0] is batch_size
                if save_result:
                    save_predict(full_prob[index], None, name[index], args.dataset, args.save_seg_dir,
                                 output_grey=True, output_color=False, gt_color=False)

        loss, FWIoU, Miou, Miou_Noback, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_Noback = 0, 0, 0, 0, {}, 0, {}, 0, 0, {}, 0

    return loss, FWIoU, Miou, Miou_Noback, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_Noback