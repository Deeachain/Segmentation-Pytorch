# _*_ coding: utf-8 _*_
"""
Time:     2020/11/27 10:23
Author:   Cheng Ding(Deeachain)
File:     resnet.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchsummary import summary

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, out_stride, mult_grid):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if out_stride == 8:
            stride = [2, 1, 1]
        elif out_stride == 16:
            stride = [2, 2, 1]
        elif out_stride == 32:
            stride = [2, 2, 2]
        # setting resnet last layer with dilation
        if mult_grid:
            if layers[-1] == 3: # layers >= 50
                mult_grid = [2, 4, 6]
                mult_grid = [4, 8, 16]
            else:
                mult_grid = [2, 4]
                mult_grid = [4, 8]
        else:
            mult_grid = []
            
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[2], dilation=mult_grid)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=[]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if dilation != []:
            layers.append(block(self.inplanes, planes, dilation[0], stride, downsample))
        else:
            layers.append(block(self.inplanes, planes, 1, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if dilation != []:
                layers.append(block(self.inplanes, planes, dilation[i]))
            else:
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        blocks = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)

        return blocks


def resnet18(out_stride=32, mult_grid=False):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], out_stride, mult_grid)

    return model


def resnet34(out_stride=32, mult_grid=False):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], out_stride, mult_grid)
 
    return model


def resnet50(out_stride=32, mult_grid=False):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], out_stride, mult_grid)

    return model


def resnet101(out_stride=32, mult_grid=False):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], out_stride, mult_grid)
    return model


def resnet152(out_stride=32, mult_grid=False):
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], out_stride, mult_grid)
    return model


"""print layers and params of network"""
if __name__ == '__main__':
    model = resnet18(pretrained=True)
    model_dict = model.state_dict()
    # for k, v in model_dict.items():
    #     print(k, v)
    print(model)
    # summary(model, (3, 512, 512), device="cpu")
