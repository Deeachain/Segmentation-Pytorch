"""
Reference to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
Add metrics: Precision、Recall、F1-Score
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
sum(axis=0) TP+FN
sum(axis=1) TP+FP
np.diag().sum() TP+TN
"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + FN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def meanPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = TP / (TP + FP)
        Cpa = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        Mpa = np.nanmean(Cpa)  # 求各类别Cpa的平均
        return Mpa, Cpa  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率


    def meanIntersectionOverUnion(self):
        # Intersection = TP ;Union = TP + FP + FN
        # Ciou = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表

        Ciou = (intersection / np.maximum(1.0, union))   # 返回列表，其值为各个类别的Ciou
        mIoU = np.nanmean(Ciou)  # 求各类别Ciou的平均
        return mIoU, Ciou

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def precision(self):
        # precision = TP / (TP+FP)
        precision = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        return precision

    def recall(self):
        # recall = TP / (TP+FN)
        recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=0)
        return recall

    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype('int') + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

if __name__ == '__main__':
    imgPredict = np.array([0, 0, 1, 1, 2, 2]) # 可直接换成预测图片
    imgLabel = np.array([0, 0, 1, 1, 1, 2]) # 可直接换成标注图片
    metric = SegmentationMetric(3) # 3表示有3个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU, per = metric.meanIntersectionOverUnion()
    print('pa is : %f' % pa)
    print('cpa is :') # 列表
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU, per)
