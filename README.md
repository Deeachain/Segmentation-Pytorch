## 由于时间有限,代码注释没有完善好,我会长期更新,不断完善readme形成一个完整的代码引导,后期数据集也会共享,方便下载调试,谢谢大家的关注~
# 1. 效果展示
- PSPNet使用的是作者开源的源代码，训练Cityscapes数据集，Miou=0.5535604759342954，效果有待提升，时间有限单卡训练200Epoch。
```
cityscapes 19 class iou 
[0.9486701457704959, 0.7049106876005735, 0.8300539507571478, 0.3252279876366611, 0.32811147036266664, 
 0.40784611045938335, 0.4032697357010293, 0.5584767609290046, 0.8800581209068778, 0.4874004109192292,
 0.9049523717813096, 0.6553959694622065, 0.4079456052528648, 0.8765924542132393, 0.19745433079577926, 
 0.510012327134576, 0.11816104748732476, 0.34569161104616813, 0.6274179445350734]
```
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
- Ubuntu 16.04 GTX1080TI 单卡 
- python==3.6.5<br>
具体参数见requirement.txt<br>
# 3. 数据预处理
本实验可以实现cityscapes和camvid公开数据集<br>
训练演示使用的是cityscapes<br>
本人主要是用作遥感卫星图像分割,卫星图像尺寸都较大,所以需要进行切图,切分成512*512尺寸大小的图片<br>
后期会更新我的数据集-----
# 4. 模型搭建
所有的模型搭建都是在builders/model_builder.py文件下导入<br>
- [x] FCN
- [x] UNet
- [x] ENet
- [x] ESPNet
- [x] ESPNet_v2
- [x] ERFNet
- [x] DABNet
- [x] BiSeNetV2
- [x] PSPNet
- [x] DeeplabV3Plus
# 5. 训练
## 5.1 数据集准备
cityscapes的训练：gtFine中的文件需要是onehot之后的图片，总共19个类别（图片中的像素是0-18&255）。<br>
- **数据集文件结构:**
```
|--cityscapes
|    |--leftImg8bit
|       |--train
|           |--zurich
|               |--leftImg8bit*.png
|           |--*
|       |--val
|           |--zurich
|               |--leftImg8bit*.png
|           |--*
|       |--test
|           |--zurich
|               |--leftImg8bit*.png
|           |--*
|    |--gtFine
|       |--train
|           |--zurich
|               |--gtFine_labelTrainIds*.png
|           |--*
|       |--val
|           |--zurich
|               |--gtFine_labelTrainIds*.png
|           |--*
|       |--test
|           |--zurich
|               |--gtFine_labelTrainIds*.png 
```

- 修改`builders/datasets_builder.py`中第11行的数据集加载路径`data_dir`为自己的cityscapes数据集目录，我是将cityscapes数据文件夹放在
`/media/ding/Data/datasets`下, 仅需修改自己本地存放数据集的目录。`os.path.join('/media/ding/Data/datasets', dataset)`
第二个参数`dataset`不需要修改，只需将数据集文件结构改成上述结构，训练的时候传参即可自动获取。此处需注意传参的时候
`--dataset cityscapes`中的`cityscapes`需要与数据集的目录名一致

- 生成训练需要的文件路径的txt文本，`cityscapes_train_list.txt`、`cityscapes_val_list.txt`、`cityscapes_test_list.txt`
放在修改过的`data_dir`下。txt的格式如下（原图路径+分隔符(\t)+标签路径）<br>
```
leftImg8bit/train/cologne/cologne_000000_000019_leftImg8bit.png gtFine/train/cologne/cologne_000000_000019_gtFine_labelTrainIds.png
```
- 终端下:`sh train.sh`<br>
`train.sh`脚本,修改相应参数；（详细参数见train.py中的ArgumentParser()）<br>
```
主要参数：
--model              训练的模型
--dataset            训练的数据集名称（与文件目录相同）
--max_epochs         训练Epoch
--val_miou_epochs    每100个Epoch验证计算miou
--lr                 学习率
--batch_size         batch_size
```
```
python train.py --max_epochs 200 --batch_size 2 --model ENet --dataset cityscapes --optim sgd --lr 0.01
```

## 5.2 测试训练保存的所有模型好坏，终端中执行:`sh test.sh`
test.py文件是用来测试所有训练生成模型权重的好坏,在所有保存的模型权重中测试得到一个指标最高的模型。修改`test.sh`的参数，`best $True`
表示选择最优模型

```
python test.py --dataset cityscapes \
               --model ENet \
               --best $True \
               --checkpoint /media/ding/Study/graduate/code/Efficient-Segmentation-Networks/checkpoint/cityscapes/ENetbs8gpu1_train/model_300.pth
```

## 5.3 两种预测方法
#### 5.3.1 预测小图,拼接成大图
终端中执行`sh predict.sh`脚本预测小图,修改`--checkpoint`等参数，接着`outputs/concat_image.py`拼接成大图

#### 5.3.2 直接输入大图,滑动窗口进行预测（本人使用的方法）
终端中执行`sh predict_sliding.sh`脚本，通过滑动窗口重叠预测，投票决定最终预测结果，得到预测大图。<br>

```
python predict_sliding.py --dataset cityscapes \
                          --model ENet \
                          --checkpoint /media/ding/Study/graduate/code/Efficient-Segmentation-Networks/checkpoint/paris/ENetbs16gpu1_train/model_91.pth
```


