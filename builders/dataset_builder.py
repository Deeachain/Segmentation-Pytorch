import os
import pickle
from torch.utils import data
from dataset.cityscapes.cityscapes import CityscapesTrainDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet
from dataset.camvid.camvid import CamVidTrainDataSet, CamVidTrainInform
from dataset.paris.paris import ParisTrainDataSet, ParisTestDataSet, ParisTrainInform
from dataset.road.road import RoadTrainDataSet, RoadTrainInform, RoadTestDataSet


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    data_dir = os.path.join('/media/ding/Data/datasets', dataset)
    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
    val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=train_data_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=train_data_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'paris':
            dataCollect = ParisTrainInform(data_dir, 3, train_set_file=train_data_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'road':
            dataCollect = ParisTrainInform(data_dir, 2, train_set_file=train_data_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        trainLoader = data.DataLoader(
            CityscapesTrainDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        return datas, trainLoader

    elif dataset == "camvid":
        trainLoader = data.DataLoader(
            CamVidTrainDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        return datas, trainLoader

    elif dataset == "paris":
        trainLoader = data.DataLoader(
            ParisTrainDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        return datas, trainLoader

    elif dataset == "road":
        trainLoader = data.DataLoader(
            RoadTrainDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        return datas, trainLoader


def build_dataset_test(dataset, num_workers, sliding=True, none_gt=False):
    data_dir = os.path.join('/media/ding/Data/datasets', dataset)
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    dataset_list = os.path.join(data_dir, dataset + '_train_list.txt')
    if sliding:
        test_data_list = os.path.join(data_dir, dataset + '_sliding_test' + '_list.txt')
    else:
        test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'paris':
            dataCollect = ParisTrainInform(data_dir, 3, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'road':
            dataCollect = RoadTrainInform(data_dir, 2, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if none_gt:
            testLoader = data.DataLoader(
                CityscapesTestDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesValDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return testLoader

    elif dataset == "camvid":
        testLoader = data.DataLoader(
            CamVidDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return testLoader

    elif dataset == "paris":
        testLoader = data.DataLoader(
            ParisTestDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return testLoader

    elif dataset == "road":
        testLoader = data.DataLoader(
            RoadTestDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return testLoader