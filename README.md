## 项目结构调整，已经删除之前代码，调整结束会重新上传代码
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