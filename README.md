## 项目更新日志
- 2020.12.10 项目结构调整，已经删除之前代码，调整结束会重新上传代码
- 2021.04.09 重新上传代码，"V1 commit"
- 2021.04.22 更新torch distributed training
- 持续更新

# 1. 效果展示（cityscapes）
- 使用模型 DDRNet 1525张测试集,官方Miou=78.4069%

<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/average_results.png"><div align = "center">Average results</div></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-PytoWrch/blob/master/example/class_results1.png"><div align = "center">Class results</div></center></td>
    </tr>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/class_results2.png"><div align = "center">Average results</div></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/class_results3.png"><div align = "center">Class results</div></center></td>
    </tr>
</table>

- 原图和预测图对比
<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit.png"><div align = "center">origin</div></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit_gt.png"><div align = "center">gt</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit_color.png"><div align = "center">predict</div></center></td>
    </tr>
</table>

# 2. 环境安装
```pip install -r requirements.txt```<br>
实验环境： 
- Ubuntu 16.04 Nvidia-Cards >= 1
- python==3.6.5<br>
- 具体依赖安装包见requirement.txt<br>

# 3. 模型搭建
所有的模型搭建都是在builders/model_builder.py文件下导入<br>
- [x] FCN
- [x] FCN_ResNet
- [x] SegNet
- [x] UNet
- [x] BiSeNet
- [x] BiSeNetV2
- [x] PSPNet
- [x] DeepLabv3_plus
- [x] HRNet
- [x] DDRNet

# 4. 数据预处理
本项目可以实现公开数据集Cityscapes、ISPRS<br>
后期会上传数据集-----<br>
此处展示Cityscapes数据集准备：

## 4.1 首先下载数据集
原始数据集的标签灰度图onehot编码是0-32,语义分割任务真实用到的标签灰度图onehot编码是0-18,因此需要对标签进行编码。
```
原图：aachen_000000_000019_leftImg8bit
彩色标签：aachen_000000_000019_gtFine_color
灰度标签(0-18)：aachen_000000_000019_gtFine_labelTrainIds
```

- 原图、彩色标签图和灰度标签图(0-18)对比
<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/aachen_000000_000019_leftImg8bit.png"><div align = "center">***_leftImg8bit</div></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/aachen_000000_000019_gtFine_color.png"><div align = "center">***_gtFine_color</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/aachen_000000_000019_gtFine_labelTrainIds.png"><div align = "center">***_gtFine_labelTrainIds</div></center></td>
    </tr>
</table>

- 本地存储路径展示:(/data/open_data/cityscapes/)
```
data
  |--open_data
        |--cityscapes
               |--leftImg8bit
                    |--train
                        |--cologne
                        |--*******
                    |--val
                        |--*******
                    |--test
                        |--*******
               |--gtFine
                    |--train
                        |--cologne
                        |--*******
                    |--val
                        |--*******
                    |--test
                        |--*******
```

- 生成图片路径txt<br>
根据`dataset/generate_txt.py`脚本生成包含原图和标签的路径`txt`文件<br>
总共生成3个`txt`文件:`cityscapes_train_list.txt`、`cityscapes_val_list.txt`、`cityscapes_test_list.txt`,并且将三个文件复制到数据集根目录下<br>
```
data
  |--open_data
        |--cityscapes
               |--cityscapes_train_list.txt
               |--cityscapes_val_list.txt
               |--cityscapes_test_list.txt
               |--leftImg8bit
                    |--train
                        |--cologne
                        |--*******
                    |--val
                        |--*******
                    |--test
                        |--*******
               |--gtFine
                    |--train
                        |--cologne
                        |--*******
                    |--val
                        |--*******
                    |--test
                        |--*******
```

- `txt`文件内容展示如下:
```
leftImg8bit/train/cologne/cologne_000000_000019_leftImg8bit.png gtFine/train/cologne/cologne_000000_000019_gtFine_labelTrainIds.png
leftImg8bit/train/cologne/cologne_000001_000019_leftImg8bit.png gtFine/train/cologne/cologne_000001_000019_gtFine_labelTrainIds.png
..............
```

- `txt`文件格式如下:
```
原图路径+分隔符'\t'+标签路径+'\n'
```


# TODO.....
# 5. 如何训练
```
sh train.sh
```
## 5.1 参数详解
```
python -m torch.distributed.launch --nproc_per_node=2 \
                train.py --model PSPNet_res50 --out_stride 8 \
                --max_epochs 200 --val_epochs 20 --batch_size 4 --lr 0.01 --optim sgd --loss ProbOhemCrossEntropy2d \
                --base_size 768 --crop_size 768  --tile_hw_size 768,768 \
                --root '/data/open_data' --dataset cityscapes --gpus_id 1,2
```
# 6. 如何验证
```
sh predict.sh
```