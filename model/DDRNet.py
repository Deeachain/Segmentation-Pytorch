"""
This script contain three DDRNet: DDRNet23_slim、DDRNet23、DDRNet39
Reference from source code by author: https://github.com/ydhongHIT/DDRNet
"""

'''
DDRNet23_slim
'''

# import math
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# from collections import OrderedDict
#
# # BatchNorm2d = nn.SyncBatchNorm
# BatchNorm2d = nn.BatchNorm2d
# bn_mom = 0.1
#
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
#         self.downsample = downsample
#         self.stride = stride
#         self.no_relu = no_relu

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual

#         if self.no_relu:
#             return out
#         else:
#             return self.relu(out)

# class Bottleneck(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
#                                bias=False)
#         self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.no_relu = no_relu

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         if self.no_relu:
#             return out
#         else:
#             return self.relu(out)

# class DAPPM(nn.Module):
#     def __init__(self, inplanes, branch_planes, outplanes):
#         super(DAPPM, self).__init__()
#         self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale0 = nn.Sequential(
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.process1 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )
#         self.process2 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )
#         self.process3 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )
#         self.process4 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )        
#         self.compression = nn.Sequential(
#                                     BatchNorm2d(branch_planes * 5, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
#                                     )
#         self.shortcut = nn.Sequential(
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
#                                     )

#     def forward(self, x):

#         #x = self.downsample(x)
#         width = x.shape[-1]
#         height = x.shape[-2]        
#         x_list = []

#         x_list.append(self.scale0(x))
#         x_list.append(self.process1((F.interpolate(self.scale1(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[0])))
#         x_list.append((self.process2((F.interpolate(self.scale2(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[1]))))
#         x_list.append(self.process3((F.interpolate(self.scale3(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[2])))
#         x_list.append(self.process4((F.interpolate(self.scale4(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[3])))
       
#         out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
#         return out 


# class segmenthead(nn.Module):

#     def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
#         super(segmenthead, self).__init__()
#         self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
#         self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
#         self.scale_factor = scale_factor

#     def forward(self, x):
        
#         x = self.conv1(self.relu(self.bn1(x)))
#         out = self.conv2(self.relu(self.bn2(x)))

#         if self.scale_factor is not None:
#             height = x.shape[-2] * self.scale_factor
#             width = x.shape[-1] * self.scale_factor
#             out = F.interpolate(out,
#                         size=[height, width],
#                         mode='bilinear')

#         return out

# class DualResNet(nn.Module):

#     def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=False):
#         super(DualResNet, self).__init__()

#         highres_planes = planes * 2
#         self.augment = augment

#         self.conv1 =  nn.Sequential(
#                           nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
#                           BatchNorm2d(planes, momentum=bn_mom),
#                           nn.ReLU(inplace=True),
#                           nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
#                           BatchNorm2d(planes, momentum=bn_mom),
#                           nn.ReLU(inplace=True),
#                       )

#         self.relu = nn.ReLU(inplace=False)
#         self.layer1 = self._make_layer(block, planes, planes, layers[0])
#         self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

#         self.compression3 = nn.Sequential(
#                                           nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
#                                           BatchNorm2d(highres_planes, momentum=bn_mom),
#                                           )

#         self.compression4 = nn.Sequential(
#                                           nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
#                                           BatchNorm2d(highres_planes, momentum=bn_mom),
#                                           )

#         self.down3 = nn.Sequential(
#                                    nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
#                                    BatchNorm2d(planes * 4, momentum=bn_mom),
#                                    )

#         self.down4 = nn.Sequential(
#                                    nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
#                                    BatchNorm2d(planes * 4, momentum=bn_mom),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
#                                    BatchNorm2d(planes * 8, momentum=bn_mom),
#                                    )

#         self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

#         self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

#         self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

#         self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

#         self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

#         if self.augment:
#             self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)            

#         self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
#             )

#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             if i == (blocks-1):
#                 layers.append(block(inplanes, planes, stride=1, no_relu=True))
#             else:
#                 layers.append(block(inplanes, planes, stride=1, no_relu=False))

#         return nn.Sequential(*layers)


#     def forward(self, x):
#         B, C, H, W = x.shape
#         width_output = x.shape[-1] // 8 
#         height_output = x.shape[-2] // 8 
#         layers = []

#         x = self.conv1(x)

#         x = self.layer1(x)
#         layers.append(x)

#         x = self.layer2(self.relu(x))
#         layers.append(x)
  
#         x = self.layer3(self.relu(x))
#         layers.append(x)
#         x_ = self.layer3_(self.relu(layers[1]))

#         x = x + self.down3(self.relu(x_))

#         x_ = x_ + F.interpolate(
#                         self.compression3(self.relu(layers[2])),
#                         size=[height_output, width_output],
#                         # scale_factor = 2,
#                         mode='bilinear')
#         if self.augment:
#             temp = x_

#         x = self.layer4(self.relu(x))
#         layers.append(x)
#         x_ = self.layer4_(self.relu(x_))

#         x = x + self.down4(self.relu(x_))
#         x_ = x_ + F.interpolate(
#                         self.compression4(self.relu(layers[3])),
#                         size=[height_output, width_output],
#                         # scale_factor = 2,
#                         mode='bilinear')

#         x_ = self.layer5_(self.relu(x_))
#         x = F.interpolate(
#                         self.spp(self.layer5(self.relu(x))),
#                         size=[height_output, width_output],
#                         # scale_factor = 2,
#                         mode='bilinear')

#         x_ = self.final_layer(x + x_)

#         x_ = F.interpolate(
#                         x_,
#                         size=[H, W],
#                         # scale_factor = 2,
#                         mode='bilinear')

#         if self.augment: 
#             x_extra = self.seghead_extra(temp)
#             x_extra = F.interpolate(
#                         x_extra,
#                         size=[H, W],
#                         # scale_factor = 2,
#                         mode='bilinear')
#             return x_, x_extra
#         else:
#             return x_      

# def DDRNet(pretrained=True, num_classes=19):
#     model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, planes=32, spp_planes=128, head_planes=64, augment=True)
#     if pretrained:
#         checkpoint = torch.load('/data/dingcheng/segment/DDRNet23s_imagenet.pth', map_location='cpu') 
              
#         # new_state_dict = OrderedDict()
#         # for k, v in checkpoint['state_dict'].items():
#         #     name = k[7:]  
#         #     new_state_dict[name] = v

#         # model_dict.update(new_state_dict)
#         # model.load_state_dict(model_dict)
        
#         model.load_state_dict(checkpoint, strict = False)
#     return model

# def get_seg_model(cfg, **kwargs):

#     model = DDRNet(pretrained=False)
#     return model





'''
DDRNet23
'''

# import math
# import torch
# import numpy as np 
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# from collections import OrderedDict

# BatchNorm2d = nn.BatchNorm2d
# bn_mom = 0.1


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
#         self.downsample = downsample
#         self.stride = stride
#         self.no_relu = no_relu

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual

#         if self.no_relu:
#             return out
#         else:
#             return self.relu(out)

# class Bottleneck(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion,
#                                momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.no_relu = no_relu

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         if self.no_relu:
#             return out
#         else:
#             return self.relu(out)

# class DAPPM(nn.Module):
#     def __init__(self, inplanes, branch_planes, outplanes):
#         super(DAPPM, self).__init__()
#         self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale0 = nn.Sequential(
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.process1 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )
#         self.process2 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )
#         self.process3 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )
#         self.process4 = nn.Sequential(
#                                     BatchNorm2d(branch_planes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#                                     )        
#         self.compression = nn.Sequential(
#                                     BatchNorm2d(branch_planes * 5, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
#                                     )
#         self.shortcut = nn.Sequential(
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
#                                     )

#     def forward(self, x):

#         #x = self.downsample(x)
#         width = x.shape[-1]
#         height = x.shape[-2]        
#         x_list = []

#         x_list.append(self.scale0(x))
#         x_list.append(self.process1((F.interpolate(self.scale1(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[0])))
#         x_list.append((self.process2((F.interpolate(self.scale2(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[1]))))
#         x_list.append(self.process3((F.interpolate(self.scale3(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[2])))
#         x_list.append(self.process4((F.interpolate(self.scale4(x),
#                         size=[height, width],
#                         mode='bilinear')+x_list[3])))
       
#         out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
#         return out 

# class segmenthead(nn.Module):

#     def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
#         super(segmenthead, self).__init__()
#         self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
#         self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
#         self.scale_factor = scale_factor

#     def forward(self, x):
        
#         x = self.conv1(self.relu(self.bn1(x)))
#         out = self.conv2(self.relu(self.bn2(x)))

#         if self.scale_factor is not None:
#             height = x.shape[-2] * self.scale_factor
#             width = x.shape[-1] * self.scale_factor
#             out = F.interpolate(out,
#                         size=[height, width],
#                         mode='bilinear')

#         return out

# class DualResNet(nn.Module):

#     def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=False):
#         super(DualResNet, self).__init__()

#         highres_planes = planes * 2
#         self.augment = augment

#         self.conv1 =  nn.Sequential(
#                           nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
#                           BatchNorm2d(planes, momentum=bn_mom),
#                           nn.ReLU(inplace=True),
#                           nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
#                           BatchNorm2d(planes, momentum=bn_mom),
#                           nn.ReLU(inplace=True),
#                       )

#         self.relu = nn.ReLU(inplace=False)
#         self.layer1 = self._make_layer(block, planes, planes, layers[0])
#         self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
#         self.layer3_1 = self._make_layer(block, planes * 2, planes * 4, layers[2] // 2, stride=2)
#         self.layer3_2 = self._make_layer(block, planes * 4, planes * 4, layers[2] // 2)
#         self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

#         self.compression3_1 = nn.Sequential(
#                                           nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
#                                           BatchNorm2d(highres_planes, momentum=bn_mom),
#                                           )

#         self.compression3_2 = nn.Sequential(
#                                           nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
#                                           BatchNorm2d(highres_planes, momentum=bn_mom),
#                                           )

#         self.compression4 = nn.Sequential(
#                                           nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
#                                           BatchNorm2d(highres_planes, momentum=bn_mom),
#                                           )

#         self.down3_1 = nn.Sequential(
#                                    nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
#                                    BatchNorm2d(planes * 4, momentum=bn_mom),
#                                    )

#         self.down3_2 = nn.Sequential(
#                                    nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
#                                    BatchNorm2d(planes * 4, momentum=bn_mom),
#                                    )

#         self.down4 = nn.Sequential(
#                                    nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
#                                    BatchNorm2d(planes * 4, momentum=bn_mom),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
#                                    BatchNorm2d(planes * 8, momentum=bn_mom),
#                                    )

#         self.layer3_1_ = self._make_layer(block, planes * 2, highres_planes, layers[2] // 2)

#         self.layer3_2_ = self._make_layer(block, highres_planes, highres_planes, layers[2] // 2)

#         self.layer4_ = self._make_layer(block, highres_planes, highres_planes, layers[3])

#         self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

#         self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

#         self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

#         if self.augment:
#             self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

#         self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
#             )

#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             if i == (blocks-1):
#                 layers.append(block(inplanes, planes, stride=1, no_relu=True))
#             else:
#                 layers.append(block(inplanes, planes, stride=1, no_relu=False))

#         return nn.Sequential(*layers)


#     def forward(self, x):
#         B, C, H, W = x.shape
#         width_output = x.shape[-1] // 8
#         height_output = x.shape[-2] // 8
#         layers = []

#         x = self.conv1(x)

#         x = self.layer1(x)
#         layers.append(x)

#         x = self.layer2(self.relu(x))
#         layers.append(x)
  
#         x = self.layer3_1(self.relu(x))
#         layers.append(x)
#         x_ = self.layer3_1_(self.relu(layers[1]))
#         x = x + self.down3_1(self.relu(x_))
#         x_ = x_ + F.interpolate(
#                         self.compression3_1(self.relu(layers[2])),
#                         size=[height_output, width_output],
#                         mode='bilinear')

#         x = self.layer3_2(self.relu(x))
#         layers.append(x)
#         x_ = self.layer3_2_(self.relu(x_))
#         x = x + self.down3_2(self.relu(x_))
#         x_ = x_ + F.interpolate(
#                         self.compression3_2(self.relu(layers[3])),
#                         size=[height_output, width_output],
#                         mode='bilinear')

#         temp = x_

#         x = self.layer4(self.relu(x))
#         layers.append(x)
#         x_ = self.layer4_(self.relu(x_))
#         x = x + self.down4(self.relu(x_))
#         x_ = x_ + F.interpolate(
#                         self.compression4(self.relu(layers[4])),
#                         size=[height_output, width_output],
#                         mode='bilinear')

#         x_ = self.layer5_(self.relu(x_))
#         x = F.interpolate(
#                         self.spp(self.layer5(self.relu(x))),
#                         size=[height_output, width_output],
#                         mode='bilinear')

#         x_ = self.final_layer(x + x_)

#         x_ = F.interpolate(
#                         x_,
#                         size=[H, W],
#                         # scale_factor = 2,
#                         mode='bilinear')
#         if self.augment: 
#             x_extra = self.seghead_extra(temp)
#             return [x_, x_extra]
#         else:
#             return x_      

# def DDRNet(pretrained=False, num_classes=19):
#     model = DualResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, planes=64, spp_planes=128, head_planes=256, augment=False)
#     if pretrained:
#         checkpoint = torch.load('/data/dingcheng/segment/DDRNet23_imagenet.pth', map_location='cpu')
#         '''       
#         new_state_dict = OrderedDict()
#         for k, v in checkpoint['state_dict'].items():
#             name = k[7:]  
#             new_state_dict[name] = v
#         #model_dict.update(new_state_dict)
#         #model.load_state_dict(model_dict)
#         '''
#         model.load_state_dict(checkpoint, strict = False)
#     return model


# def get_seg_model():

#     model = DDRNet(pretrained=False)
#     # print(model)
#     return model



'''
DDRNet39
'''

import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

BatchNorm2d = nn.SyncBatchNorm
# BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

        return out

class DualResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=False):
        super(DualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3_1 = self._make_layer(block, planes * 2, planes * 4, layers[2] // 2, stride=2)
        self.layer3_2 = self._make_layer(block, planes * 4, planes * 4, layers[2] // 2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3_1 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression3_2 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.down3_1 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down3_2 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_1_ = self._make_layer(block, planes * 2, highres_planes, layers[2] // 2)

        self.layer3_2_ = self._make_layer(block, highres_planes, highres_planes, layers[2] // 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, layers[3])

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def forward(self, x):
        B, C, H, W = x.shape
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)
  
        x = self.layer3_1(self.relu(x))
        layers.append(x)
        x_ = self.layer3_1_(self.relu(layers[1]))
        x = x + self.down3_1(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3_1(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x = self.layer3_2(self.relu(x))
        layers.append(x)
        x_ = self.layer3_2_(self.relu(x_))
        x = x + self.down3_2(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3_2(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear')

        temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))
        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[4])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.final_layer(x + x_)

        x_ = F.interpolate(x_, size=[H, W], mode='bilinear')

        if self.augment: 
            x_extra = self.seghead_extra(temp)
            return [x_, x_extra]
        else:
            return x_      

def DDRNet(pretrained=True, num_classes=19):
    model = DualResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, planes=64, spp_planes=128, head_planes=256, augment=False)
    if pretrained:
        checkpoint = torch.load('/data/dingcheng/segment/DDRNet39_imagenet.pth', map_location='cpu')
        '''       
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  
            new_state_dict[name] = v
        #model_dict.update(new_state_dict)
        #model.load_state_dict(model_dict)
        '''
        model.load_state_dict(checkpoint, strict = False)
    return model


def get_seg_model(cfg, **kwargs):

    model = DDRNet(pretrained=True, num_classes=19)
    return model



if __name__ =='__main__':
    get_seg_model()