import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.losses.lovasz_losses import lovasz_softmax
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import NLLLoss2d
from collections import OrderedDict

__all__ = ["CrossEntropyLoss2d", "CrossEntropyLoss2dLabelSmooth",
           "FocalLoss2d", "LDAMLoss", "ProbOhemCrossEntropy2d",
           "LovaszSoftmax"]


class LabelSmoothing(nn.Module):
    '''
    Description: 借鉴标签平滑的思想，针对样本中的预假设的 `hard sample`（图像边缘、不同类别交界） 进行标签平滑；
                 平滑因子可指定 smoothing 固定，或在训练过程中，在图像边缘、类间交界设置一定大小过渡带，统计过渡带
                 内像素 `hard sample` 比例动态调整。
    Args (type):
        win_size (int): 过渡带窗口大小；
        num_classes (int): 总类别数目，本次实验类别数为5；
        smoothing (float): 默认值为0.1，若指定 fix_smoothing ，则固定训练过程固定平滑因子为 smoothing。 
    '''
    def __init__(self,win_size=11,num_classes=6,smoothing=0.1,fix_smoothing=False):
        super(LabelSmoothing, self).__init__()
        self.fix_smoothing = fix_smoothing
        assert (win_size%2) == 1
        self.smoothing = smoothing /(num_classes-1)
        self.win_size = win_size
        self.num_classes = num_classes
        
        self.find_edge_Conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=win_size,padding=(win_size-1)//2,stride=1,bias=False)
        self.find_edge_Conv.weight.requires_grad = False
        new_state_dict = OrderedDict()
        weight = torch.zeros(1,1,win_size,win_size)
        weight = weight -1
        weight[:,:,win_size//2,win_size//2] = win_size*win_size - 1
        new_state_dict['weight'] = weight
        self.find_edge_Conv.load_state_dict(new_state_dict)


    def to_categorical(self,y,alpha=0.05,num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = (1-alpha) + (alpha/self.num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0,3,1,2)
        return categorical
        

    def forward(self, x, target):
        assert x.size(1) == self.num_classes
        log_p = nn.functional.log_softmax(x,dim=1)
        
        if not self.fix_smoothing:
            self.find_edge_Conv.cuda(device=target.device)
            edge_mask = self.find_edge_Conv(target)
            edge_mask = edge_mask.data.cpu().numpy()
            edge_mask[edge_mask!=0] = 1
            self.smoothing = np.mean(edge_mask)
            if self.smoothing > 0.2:
                self.smoothing = 0.2
        
        target = target.squeeze(dim=1)
        target = target.data.cpu().numpy()
        onehot_mask = self.to_categorical(target,0,num_classes=self.num_classes)
        onehot_mask = onehot_mask*(1-edge_mask)
        softlabel_mask = self.to_categorical(target,alpha=self.smoothing,num_classes=self.num_classes)
        softlabel_mask = softlabel_mask*edge_mask
        onehot_mask = torch.from_numpy(onehot_mask).cuda(device=log_p.device).float()
        softlabel_mask = torch.from_numpy(softlabel_mask).cuda(device=log_p.device).float()
        loss = torch.sum(onehot_mask*log_p+softlabel_mask*log_p,dim=1).mean()
        return -loss


class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, ignore_label=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()

        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(output, target)


class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
    """
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    """

    def __init__(self, weight=None, ignore_label=255, epsilon=0.1, reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        n_classes = output.size(1)
        targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes

        return self.nll_loss(output, targets)


"""
https://arxiv.org/abs/1708.02002
# Credit to https://github.com/clcarwin/focal_loss_pytorch
"""


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):

        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


"""
https://arxiv.org/pdf/1906.07413.pdf
"""


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


# # Adapted from OCNet Repository (https://github.com/PkuRainBow/OCNet)
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, weight=None):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if weight is not None:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                                 weight=weight,
                                                 ignore_index=ignore_label)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                                 ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            # print('Labels: {}'.format(num_valid))
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)  #
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


# class ProbOhemCrossEntropy2d(nn.Module):
#     def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
#                  down_ratio=1, weight=None):
#         super(ProbOhemCrossEntropy2d, self).__init__()
#         self.ignore_label = ignore_label
#         self.thresh = float(thresh)
#         self.min_kept = int(min_kept)
#         self.down_ratio = down_ratio
#         if weight is not None:
#             self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
#                                                        weight=weight,
#                                                        ignore_index=ignore_label)
#         else:
#             self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
#                                                        ignore_index=ignore_label)
#
#     def forward(self, pred, target):
#         b, c, h, w = pred.size()
#         target = target.view(-1)
#         valid_mask = target.ne(self.ignore_label)
#         target = target * valid_mask.long()
#         num_valid = valid_mask.sum()
#
#         prob = F.softmax(pred, dim=1)
#         prob = (prob.transpose(0, 1)).reshape(c, -1)
#
#         if self.min_kept > num_valid:
#             logger.info('Labels: {}'.format(num_valid))
#         elif num_valid > 0:
#             prob = prob.masked_fill_(1 - valid_mask, 1)
#             mask_prob = prob[
#                 target, torch.arange(len(target), dtype=torch.long)]
#             threshold = self.thresh
#             if self.min_kept > 0:
#                 _, index = torch.sort(mask_prob)
#                 threshold_index = index[min(len(index), self.min_kept) - 1]
#                 if mask_prob[threshold_index] > self.thresh:
#                     threshold = mask_prob[threshold_index]
#                 kept_mask = mask_prob.le(threshold)
#                 target = target * kept_mask.long()
#                 valid_mask = valid_mask * kept_mask
#                 # logger.info('Valid Mask: {}'.format(valid_mask.sum()))
#
#         target = target.masked_fill_(1 - valid_mask, self.ignore_label)
#         target = target.view(b, h, w)
#
#         return self.criterion(pred, target)

# class-balanced loss
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None, num_class=0):

        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        if weight is not None:
            print('target size {}'.format(target.shape))
            freq = np.zeros(num_class)
            for k in range(num_class):
                mask = (target[:, :, :] == k)
                freq[k] = torch.sum(mask)
                print('{}th frequency {}'.format(k, freq[k]))
            weight = freq / np.sum(freq)
            print(weight)
            weight = torch.FloatTensor(weight)
            print('Online class weight: {}'.format(weight))

        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_label)
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = criterion(predict, target)
        return loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
