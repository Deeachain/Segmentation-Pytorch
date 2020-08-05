import cv2
import numpy as np
from tqdm import tqdm
from glob import glob


class DatasetsInformation:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, classes=32, normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)

        for i in range(self.classes):
            self.classWeights[i] = 0.1 / (np.log(self.normVal + normHist[i]))
        return histogram, normHist * 100, self.classWeights

    def readWholeTrainSet(self, image, label, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data

        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0

        for index, label_file in tqdm(enumerate(label)):
            img_file = image[index]

            label_img = cv2.imread(label_file, 0)
            unique_values = np.unique(label_img)
            max_val = max(unique_values)
            min_val = min(unique_values)

            max_val_al = max(max_val, max_val_al)
            min_val_al = min(min_val, min_val_al)

            if train_flag == True:
                hist = np.histogram(label_img, self.classes, range=(0, self.classes - 1))
                global_hist += hist[0]

                rgb_img = cv2.imread(img_file)
                self.mean[0] += np.mean(rgb_img[:, :, 0])
                self.mean[1] += np.mean(rgb_img[:, :, 1])
                self.mean[2] += np.mean(rgb_img[:, :, 2])

                self.std[0] += np.std(rgb_img[:, :, 0])
                self.std[1] += np.std(rgb_img[:, :, 1])
                self.std[2] += np.std(rgb_img[:, :, 2])

            else:
                print("we can only collect statistical information of train set, please check")

            if max_val > (self.classes - 1) or min_val < 0:
                print('Labels can take value between 0 and number of classes.')
                print('Some problem with labels. Please check. label_set:', unique_values)
                print('Label Image ID: ' + label_file)
            no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return self.mean, self.std, self.compute_class_weights(global_hist)


if __name__ == '__main__':
    image = glob('/media/ding/Data/datasets/cityscapes/leftImg8bit/train/*/*.png')
    label = glob('/media/ding/Data/datasets/cityscapes/gtFine/train/*/*labelTrainIds.png')
    class_num = 19

    info = DatasetsInformation(classes=class_num)
    out = info.readWholeTrainSet(image=image, label=label, train_flag=True)

    np.set_printoptions(suppress=True)
    print('Std is \n', out[0])  # 计算数据集图片的均值
    print('\nMean is \n', out[1])  # 计算数据集图片的均值
    print('\nPerClass Count is \n', np.array(out[2][0], dtype=int))  # 计算每个类别的个数，返回值是一个列表，列表的长度是雷和个数
    print('\nPercentPerClass is (%)\n', out[2][1])  # 计算每个类别所占比例百分比
    print('\nCalcClassWeight is \n', out[2][2])  # 计算类别的权重

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))
    # plt.rcParams['font.sans-serif'] = ['SimSun']
    # plt.rcParams['axes.unicode_minus'] = False
    X = np.arange(0, class_num)
    Y = out[2][1]
    plt.bar(x=X, height=Y, color="c", width=0.8)
    plt.xticks(X, fontsize=20)  # 标注横坐标的类别名
    plt.yticks(fontsize=20)  # 标注纵坐标
    for x, y in zip(X, Y):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=18)  # 使用matplotlib画柱状图并标记数字
    plt.xlabel("Class", fontsize=32)
    plt.ylabel("Percent(%)", fontsize=32)
    plt.title("Category scale distribution", fontsize=36)
    plt.savefig('Percent.png')
    # plt.show()