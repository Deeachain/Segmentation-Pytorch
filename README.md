## 项目更新日志

- 2020.12.10 项目结构调整，已经删除之前代码，调整结束会重新上传代码
- 2021.04.09 重新上传代码，"V1 commit"

# 1. 效果展示
- DDRNet Miou=72.79%
```
+--------------------------------------------------------------------+
|                         Validation results                         |
+-------------+---------------+--------+-----------+--------+--------+
| label_index |   label_name  |  IoU   | Precision | Recall |   F1   |
+-------------+---------------+--------+-----------+--------+--------+
|      0      |      road     | 0.9877 |   0.9932  | 0.9944 | 0.9938 |
|      1      |    sidewalk   | 0.807  |    0.89   | 0.8964 | 0.8932 |
|      2      |    building   | 0.9182 |   0.9604  | 0.9543 | 0.9573 |
|      3      |      wall     | 0.4258 |   0.4785  | 0.7946 | 0.5973 |
|      4      |     fence     | 0.5135 |   0.664   | 0.6937 | 0.6785 |
|      5      |      pole     | 0.5758 |   0.6846  | 0.7838 | 0.7308 |
|      6      | traffic light | 0.694  |   0.804   | 0.8353 | 0.8194 |
|      7      |  traffic sign | 0.7425 |   0.8371  | 0.868  | 0.8522 |
|      8      |   vegetation  | 0.9317 |   0.9707  | 0.9586 | 0.9646 |
|      9      |    terrain    | 0.604  |   0.7055  | 0.8077 | 0.7531 |
|      10     |      sky      | 0.9525 |   0.9786  | 0.9728 | 0.9757 |
|      11     |     person    | 0.8068 |   0.9142  | 0.8729 | 0.8931 |
|      12     |     rider     | 0.6131 |   0.7588  | 0.7615 | 0.7602 |
|      13     |      car      | 0.932  |   0.9673  | 0.9623 | 0.9648 |
|      14     |     truck     | 0.6948 |   0.8009  | 0.8398 | 0.8199 |
|      15     |      bus      | 0.7804 |   0.9198  | 0.8374 | 0.8767 |
|      16     |     train     | 0.5559 |   0.5925  | 0.8998 | 0.7145 |
|      17     |   motorcycle  | 0.5718 |   0.6902  | 0.7691 | 0.7275 |
|      18     |    bicycle    | 0.7221 |   0.8675  | 0.8117 | 0.8387 |
+-------------+---------------+--------+-----------+--------+--------+
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
本项目可以实现cityscapes公开数据集<br>
后期会上传数据集-----
# 4. 模型搭建
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
