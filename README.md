## :rocket: If it helps you, click a star! :star: ##
## Update log
- 2020.12.10 Project structure adjustment, the previous code has been deleted, the adjustment will be re-uploaded code
- 2021.04.09 Re-upload the code, "V1 Commit"
- 2021.04.22 update torch distributed training
- Ongoing update .....

# 1. Display (Cityscapes)
- Using model DDRNet 1525 test sets, official MIOU =78.4069%

<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/average_results.png"><div align = "center">Average results</div></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/class_results1.png"><div align = "center">Class results1</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/class_results2.png"><div align = "center">Class results2</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/class_results3.png"><div align = "center">Class results3</div></center></td>
    </tr>
</table>

- Comparison of the original and predicted images
<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit.png"><div align = "center">origin</div></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit_gt.png"><div align = "center">label</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/lindau_000000_000019_leftImg8bit_color.png"><div align = "center">predict</div></center></td>
    </tr>
</table>

# 2. Install
```pip install -r requirements.txt```<br>
Experimental environment:
- Ubuntu 16.04 Nvidia-Cards >= 1
- python==3.6.5<br>
- See Dependency Installation Package for details in requirement.txt<br>

# 3. Model
All the modeling is done in `builders/model_builder.py`<br>
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

| Model| Backbone| Val mIoU | Test mIoU | Imagenet Pretrain| Pretrained Model |
| :--- | :---: |:---: |:---:|:---:|:---:|
| PSPNet | ResNet 50 | 76.54% | - | √ | [PSPNet](https://drive.google.com/file/d/10T321s62xDZQJUR3k0H-l64smYW0QAxN/view?usp=sharing) |
| DeeplabV3+ | ResNet 50 | 77.78% | - | √ | [DeeplabV3+](https://drive.google.com/file/d/1xP7HQwFcXAPuoL_BCYdghOBnEJIxNE-T/view?usp=sharing) |
| DDRNet23_slim | - |  |  | [DDRNet23_slim_imagenet](https://drive.google.com/file/d/1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4/view) | |
| DDRNet23 | - |  |  | [DDRNet23_imagenet](https://drive.google.com/file/d/1VoUsERBeuCaiuQJufu8PqpKKtGvCTdug/view) | |
| DDRNet39 | - | 79.63% | - | [DDRNet39_imagenet](https://drive.google.com/file/d/122CMx6DZBaRRf-dOHYwuDY9vG0_UQ10i/view) | [DDRNet39](https://drive.google.com/file/d/1-poQsQzXqGl2d2ILXRhWgQH452MUTX5y/view?usp=sharing) |
Updating more model.......

# 4. Data preprocessing
This project enables you to expose data sets: `Cityscapes`、`ISPRS`<br>
The data set is uploaded later .....<br>
**Cityscapes data set preparation is shown here:**

## 4.1 Download the dataset
Download the dataset from the link on the website, You can get `*leftImg8bit.png` suffix of original image under folder `leftImg8bit`, 
`a) *color.png`、`b) *labelIds.png`、`c) *instanceIds.png` suffix of fine labeled image under folder `gtFine`.
```
*leftImg8bit.png          : the origin picture
a) *color.png             : the class is encoded by its color
b) *labelIds.png          : the class is encoded by its ID
c) *instanceIds.png       : the class and the instance are encoded by an instance ID
```
## 4.2 Onehot encoding of label image
The real label gray scale image Onehot encoding used by the semantic segmentation task is 0-18, so the label needs to be encoded. 
Using scripts `dataset/cityscapes/cityscapes_scripts/process_cityscapes.py`
to process the image and get the result `*labelTrainIds.png`. 
`process_cityscapes.py` usage: Modify 486 lines `Cityscapes_path'is the path to store your own data.

- Comparison of original image, color label image and gray label image (0-18)
<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/aachen_000000_000019_leftImg8bit.png"><div align = "center">***_leftImg8bit</div></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/aachen_000000_000019_gtFine_color.png"><div align = "center">***_gtFine_color</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/Segmentation-Pytorch/blob/master/example/aachen_000000_000019_gtFine_labelTrainIds.png"><div align = "center">***_gtFine_labelTrainIds</div></center></td>
    </tr>
</table>

- Local storage path display `/data/open_data/cityscapes/`:
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

## 4.3 Generate image path
- Generate a txt containing the image path<br>
Use script `dataset/generate_txt.py` to generate the path `txt` file containing the original image and labels. 
A total of 3 `txt` files will be generated: `cityscapes_train_list.txt`、`cityscapes_val_list.txt`、
`cityscapes_test_list.txt`, and copy the three files to the dataset root directory.<br>
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

- The contents of the `txt` are shown as follows:
```
leftImg8bit/train/cologne/cologne_000000_000019_leftImg8bit.png gtFine/train/cologne/cologne_000000_000019_gtFine_labelTrainIds.png
leftImg8bit/train/cologne/cologne_000001_000019_leftImg8bit.png gtFine/train/cologne/cologne_000001_000019_gtFine_labelTrainIds.png
..............
```

- The format of the `txt` are shown as follows:
```
origin image path + the separator '\t' + label path +  the separator '\n'
```


# TODO.....
# 5. How to train
```
sh train.sh
```
## 5.1 Parameters
```
python -m torch.distributed.launch --nproc_per_node=2 \
                train.py --model PSPNet_res50 --out_stride 8 \
                --max_epochs 200 --val_epochs 20 --batch_size 4 --lr 0.01 --optim sgd --loss ProbOhemCrossEntropy2d \
                --base_size 768 --crop_size 768  --tile_hw_size 768,768 \
                --root '/data/open_data' --dataset cityscapes --gpus_id 1,2
```
# 6. How to validate
```
sh predict.sh
```