import os
import pickle
import pandas as pd
from torch.utils import data
from dataset.cityscapes.cityscapes import CityscapesTrainDataSet, CityscapesTrainInform, CityscapesValDataSet, \
    CityscapesTestDataSet
from dataset.paris.paris import ParisTrainDataSet, ParisTestDataSet, ParisTrainInform
from dataset.isprs.isprs import IsprsTrainDataSet, IsprsValDataSet, IsprsTestDataSet, IsprsTrainInform


def build_dataset_train(root, dataset, base_size, crop_size, batch_size, random_scale, num_workers):
    data_dir = os.path.join(root, dataset)
    train_data_list = os.path.join(data_dir, dataset + '_' + 'train_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=train_data_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'paris':
            dataCollect = ParisTrainInform(data_dir, 3, train_set_file=train_data_list,
                                           inform_data_file=inform_data_file)
        elif dataset == 'postdam' or dataset == 'vaihingen':
            dataCollect = IsprsTrainInform(data_dir, 6, train_set_file=train_data_list,
                                           inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        trainLoader = data.DataLoader(
            CityscapesTrainDataSet(data_dir, train_data_list, base_size=base_size, crop_size=crop_size,
                                   mean=datas['mean'], std=datas['std'], ignore_label=255),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=False, drop_last=True)

        return datas, trainLoader

    elif dataset == "paris":
        trainLoader = data.DataLoader(
            ParisTrainDataSet(data_dir, train_data_list, scale=random_scale, crop_size=crop_size,
                                   mean=datas['mean'], std=datas['std'], ignore_label=255),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=False, drop_last=True)

        return datas, trainLoader

    elif dataset == "postdam" or dataset == 'vaihingen':
        trainLoader = data.DataLoader(
            IsprsTrainDataSet(data_dir, train_data_list, base_size=base_size, crop_size=crop_size,
                                   mean=datas['mean'], std=datas['std'], ignore_label=255),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=False, drop_last=True)

        return datas, trainLoader


def build_dataset_test(root, dataset, crop_size, batch_size, num_workers=4, mode='whole', gt=False):
    # data_dir = os.path.join('/data/dingcheng', dataset)
    data_dir = os.path.join(root, dataset)
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    train_data_list = os.path.join(data_dir, dataset + '_train_list.txt')
    if mode == 'whole':
        test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
    else:
        test_data_list = os.path.join(data_dir, dataset + '_test_sliding' + '_list.txt')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=train_data_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'paris':
            dataCollect = ParisTrainInform(data_dir, 3, train_set_file=train_data_list,
                                           inform_data_file=inform_data_file)
        elif dataset == 'postdam' or dataset == 'vaihingen':
            dataCollect = IsprsTrainInform(data_dir, 6, train_set_file=train_data_list,
                                           inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        datas = pickle.load(open(inform_data_file, "rb"))

    class_dict_df = pd.read_csv(os.path.join('./dataset', dataset, 'class_map.csv'))
    if dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if gt:
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesValDataSet(data_dir, test_data_list, crop_size=crop_size, mean=datas['mean'],
                                     std=datas['std'], ignore_label=255),
                batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=False)
        else:
            test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesTestDataSet(data_dir, test_data_list, crop_size=crop_size, mean=datas['mean'],
                                      std=datas['std'], ignore_label=255),
                batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=False)
        return testLoader, class_dict_df

    elif dataset == "paris":
        testLoader = data.DataLoader(
            ParisTestDataSet(data_dir, test_data_list, mean=datas['mean'], std=datas['std'], ignore_label=255),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

        return testLoader, class_dict_df

    elif dataset == "postdam" or dataset == 'vaihingen':
        if mode == 'whole':
            testLoader = data.DataLoader(
                IsprsValDataSet(data_dir, test_data_list, crop_size=crop_size, mean=datas['mean'], std=datas['std'],
                ignore_label=255), batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                pin_memory=False, drop_last=False)
        else:
            testLoader = data.DataLoader(
                IsprsTestDataSet(data_dir, test_data_list, crop_size=crop_size, mean=datas['mean'], std=datas['std'],
                ignore_label=255), batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                pin_memory=False, drop_last=False)
        return testLoader, class_dict_df
