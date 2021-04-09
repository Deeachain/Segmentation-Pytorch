import os.path as osp
import numpy as np
import cv2
from torch.utils import data
import pickle
from utils.img_utils import random_mirror, normalize, random_scale, generate_random_crop_pos, random_crop_pad_to_shape


class ParisTrainDataSet(data.Dataset):
    """ 
       ParisTrainDataSet is employed to load train set
       Args:
        root: the Paris dataset path,
        list_path: Paris_train_list.txt, include partial path
    """

    def __init__(self, root='', list_path='', max_iters=None, scale=True, crop_size=(512, 512), mean=(128, 128, 128),
                 std=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            image_name = name.strip().split()[0].strip().split('/')[-1].split('.')[0]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of train dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        size = image.shape
        name = datafiles["name"]

        image = normalize(image, np.array(self.mean), np.array(self.std))
        image, label = random_mirror(image, label)

        if self.scale:
            image, label, scale = random_scale(image, label, [0.75, 1, 1.25, 1.5, 1.75, 2.0])
            crop_pos = generate_random_crop_pos(image.shape[:2], self.crop_size)
            image, _ = random_crop_pad_to_shape(image, crop_pos, self.crop_size, 0)
            label, _ = random_crop_pad_to_shape(label, crop_pos, self.crop_size, 255)

        image = image.transpose(2, 0, 1)

        return image.copy(), label.copy(), np.array(size), name


class ParisTestDataSet(data.Dataset):
    """ 
       ParisDataSet is employed to load test set
       Args:
        root: the Paris dataset path,
        list_path: Paris_test_list.txt, include partial path
    """

    def __init__(self, root='', list_path='', mean=(128, 128, 128), std=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split('\t')[0])
            label_file = osp.join(self.root, name.split('\t')[1])
            image_name = name.strip().split()[0].strip().split('/')[-1].split('.')[0]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of validation dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        size = image.shape
        name = datafiles["name"]

        image = normalize(image, np.array(self.mean), np.array(self.std))

        image = image.transpose(2, 0, 1)

        return image.copy(), label.copy(), np.array(size), name


class ParisTrainInform:
    """
        To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, data_dir='', classes=3,
                 train_set_file="", inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file
        self.label_file_list = []

    def compute_class_weights(self, histogram, labels_file_list):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        # First Mode
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

        # Second Mode
        # count = np.zeros(self.classes)
        # image_count = np.zeros(self.classes)
        # example = cv2.imread(labels_file_list[0])
        # h, w, c = example.shape
        #
        # for label in labels_file_list:
        #     data = cv2.imdecode(np.fromfile(label, dtype=np.uint8), -1)
        #
        #     for c in range(self.classes):
        #         c_sum = np.sum(data == c)  # 统计c类像素的个数
        #         count[c] += c_sum
        #         if np.sum(data == c) != 0:  # 判断该图片中是否存在第c类像素，如果存在则第c类图片个数+1
        #             image_count[c] += 1
        #
        # frequency = count / (image_count * h * w)
        # median = np.median(frequency)
        # weight = median / frequency
        # for i in range(self.classes):
        #     self.classWeights[i] = weight[i]

    def readWholeTrainSet(self, fileName, train_flag=True):
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
        with open(fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image> <Label Image>
                line_arr = line.split()
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()
                self.label_file_list.append(label_file)

                label_img = cv2.imread(label_file, 0)
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag == True:
                    hist = np.histogram(label_img, self.classes, range=(0, self.classes))
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
        self.mean /= no_files * 255
        self.std /= no_files * 255

        # compute the class imbalance information
        self.compute_class_weights(global_hist, self.label_file_list)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None
