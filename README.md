### 由于时间有限,代码注释没有完善好,我会长期更新,不断完善readme形成一个完整的代码引导,后期数据集也会共享,方便下载调试,谢谢大家的关注~
实验环境 Ubuntu 16.04 GTX1080TI python==3.6<br>
具体参数见requirement.txt<br>
## 数据预处理
本实验可以实现cityscapes和camvid公开数据集<br>
训练演示使用的是cityscapes<br>
本人主要是用作遥感卫星图像分割,卫星图像尺寸都较大,所以需要进行切图,切分成512*512尺寸大小的图片<br>
后期会更新我的数据集-----
## 模型搭建
所有的模型搭建都是在builders/model_builder.py文件下导入
## 训练
cityscapes的训练:gtFine中的文件需要是onehot之后的图片，总共19个类别（图片中的像素是0-18&255）。<br>
文件结构<br>
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
修改builders/datasets_builder.py下的数据集加载路径<br>
data_dir表示cityscapes数据的目录<br>
data_dir = os.path.join('/media/ding/Data/datasets', dataset)为自己的cityscapes数据集目录即可<br>
我是将cityscapes数据文件夹放在'/media/ding/Data/datasets'下, 仅需修改自己本地存放数据集的目录<br>
os.path.join('/media/ding/Data/datasets', dataset)第二个参数dataset不需要修改，训练的时候传参即可<br>
传参的时候注意--dataset cityscapes,cityscapes需要与数据集的目录名一致<br>

生成训练需要的文件路径的txt文本，cityscapes_train_list.txt；cityscapes_val_list.txt；cityscapes_test_list.txt放在data_dir下<br>
```
txt的格式：*leftImg8bit.png\t*labelTrainIds.png
leftImg8bit/train/cologne/cologne_000000_000019_leftImg8bit.png gtFine/train/cologne/cologne_000000_000019_gtFine_labelTrainIds.png
```


终端下:sh train.sh
train.sh脚本,修改相应参数；详细参数见train.py中的ArgumentParser()<br>
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
builders文件夹下dataset_builder.py文件的data_dir需要修改为数据集的文件夹目录

## 测试训练保存的所有模型好坏
test.py文件是用来测试所有训练生成模型权重的好坏,在所有保存的模型权重中测试得到一个指标最高的模型<br>
终端中执行: sh test.sh<br>
修改test.sh的参数<br>
best $True指选择最优模型<br>
```
python test.py --dataset cityscapes \
               --model ENet \
               --best $True \
               --checkpoint /media/ding/Study/graduate/code/Efficient-Segmentation-Networks/checkpoint/cityscapes/ENetbs8gpu1_train/model_300.pth
```
## 两种预测方法
#### 1.预测小图,拼接成大图
```
终端中执行:sh predict.sh脚本--预测小图,修改--checkpoint等参数
接着outputs/concat_image.py拼接成大图

```
#### 2.直接输入大图,滑动窗口进行预测（本人使用的方法）
终端中执行:sh predict_sliding.sh脚本--滑动窗口预测大图<br>
```
python predict_sliding.py --dataset cityscapes \
                          --model ENet \
                          --checkpoint /media/ding/Study/graduate/code/Efficient-Segmentation-Networks/checkpoint/paris/ENetbs16gpu1_train/model_91.pth
```
## 效果展示
PSPNet训练200Epoch，Miou
![](https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit_color.png)
![](https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit_gt.png)

