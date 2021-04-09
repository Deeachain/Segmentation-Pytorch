import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')#忽略结果中警告语句


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1_BN_ReLU(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def in_conv3x3(in_channels, mid_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv3x3_BN_ReLU(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )




class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
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

        out = out + residual
        out = self.relu(out)

        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x,1).unsqueeze(1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        return scale

class QYNet(nn.Module):
    def __init__(self, num_class):
        super(QYNet, self).__init__()
        self.num_inchannels = [32,64,128,256]
        self.num_blocks = [2,2,2,2]
        self.in_conv3x3 = in_conv3x3(3, 64, 64, stride=2)
        self.conv3x3_1 = conv3x3_BN_ReLU(in_channels=64, out_channels=32, stride=1)
        self.conv3x3_2 = conv3x3_BN_ReLU(in_channels=64, out_channels=64, stride=2)
        self.Basic1 = self._make_one_branch(branch_index=1, block = BasicBlock, num_blocks=self.num_blocks, num_channels=self.num_inchannels)
        self.Se1 = SELayer(self.num_inchannels[1])
        self.conv3x3_3 = conv3x3_BN_ReLU(in_channels=64, out_channels=128, stride=2)
        self.Basic2 = self._make_one_branch(branch_index=2, block = BasicBlock, num_blocks=self.num_blocks, num_channels=self.num_inchannels)
        self.Se2 = SELayer(self.num_inchannels[2])
        self.conv3x3_4 = conv3x3_BN_ReLU(in_channels=128, out_channels=256, stride=2)
        self.Basic3 = self._make_one_branch(branch_index=3, block=BasicBlock, num_blocks=self.num_blocks,
                                            num_channels=self.num_inchannels)
        self.Se3 = SELayer(self.num_inchannels[3])
        self.conv1x1_1 = conv1x1_BN_ReLU(32+64, 64)
        self.conv1x1_2 = conv1x1_BN_ReLU(64+128, 128)
        self.conv1x1_3 = conv1x1_BN_ReLU(128+256, 256)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=256,
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.SpatialGate = SpatialGate()



    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.in_conv3x3(input)
        x1 = self.conv3x3_1(x)
        x2 = self.conv3x3_2(x)
        x2_1 = F.upsample(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        xs_1 = self.SpatialGate(x1)
        x1 = xs_1 * x1
        x2_2 = self.Basic1(x2)
        x1 = torch.cat([x2_1,x1], dim=1)
        x1 = self.conv1x1_1(x1)
        x1 = self.Se1(x1)
        x2_2 = F.upsample(x2_2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        x1 = x1 + x2_2
        x3 = self.conv3x3_3(x2_2)
        x3_1 = F.upsample(x3, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        xs_2 = self.SpatialGate(x1)
        x1 = xs_2 * x1
        x3_2 = self.Basic2(x3)
        x1 = torch.cat([x3_1,x1], dim=1)
        x1 = self.conv1x1_2(x1)
        x1 = self.Se2(x1)
        x3_2 = F.upsample(x3_2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        x1 = x1 + x3_2
        x4 = self.conv3x3_4(x3_2)
        x4_1 = F.upsample(x4, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        xs_3 = self.SpatialGate(x1)
        x1 = xs_3 * x1
        x4_2 = self.Basic3(x4)
        x1 = torch.cat([x4_1, x1], dim=1)
        x1 = self.conv1x1_3(x1)
        x1 = self.Se3(x1)
        x4_2 = F.upsample(x4_2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        x1 = x1 + x4_2
        out = self.last_layer(x1)
        out = F.upsample(out, size=(input.shape[2], input.shape[3]), mode='bilinear')

        return out

# if __name__ == '__main__':
#     x = torch.randn(1,3,224,224).cuda()
#     model = HighSpeedNet(num_class=3).cuda()
#     y = model(x)
#     print(y.shape)
#     print('*'*50)


# if __name__ == '__main__':
#     import warnings
#     warnings.filterwarnings('ignore')  # 忽略结果中警告语句
#     x = torch.randn(1,3,512,512).cuda()
#     model = HighSpeedNet(num_class=3).cuda()
#     y = model(x)
#     print(y.shape)
#     import time
#     # 计算网络运行时间:
#     # start = time.time()
#     # result = model(x)
#     # end = time.time()
#     # print('time: {}'.format(end-start))
#     torch.cuda.synchronize()
#     start = time.time()
#     result = model(x)
#     torch.cuda.synchronize()
#     end = time.time()
#     print('time: {}'.format(end - start))
#     # 参数量计算，浮点数计算
#     from thop import profile
#     flops, params = profile(model, inputs=(x, ))  # profile（模型，输入数据）
#     print('flops: {}'.format(flops))
#     print('params: {}'.format(params))




