###################################################################################################
#ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1906.09826.pdf
###################################################################################################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):

        output = self.conv(input)
        output = self.bn(output)

        return F.relu(output)
		
class FCU(nn.Module):
    def __init__(self, chann, kernel_size,dropprob, dilated): 
        """
        Factorized Convolution Unit

        """     
        super(FCU,self).__init__()

        padding = int((kernel_size-1)//2) * dilated

        self.conv3x1_1 = nn.Conv2d(chann, chann, (kernel_size,1), stride=1, padding=(int((kernel_size-1)//2)*1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,kernel_size), stride=1, padding=(0,int((kernel_size-1)//2)*1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (kernel_size,1), stride=1, padding=(padding,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,kernel_size), stride=1, padding=(0,padding), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout2d(dropprob)
        
    def forward(self, input):
        residual = input
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)   

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(residual+output,inplace=True) 


class PFCU(nn.Module):
    def __init__(self,chann):
        """
        Parallel Factorized Convolution Unit

        """         
    
        super(PFCU,self).__init__()
        
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_22 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(2,0), bias=True, dilation = (2,1))
        self.conv1x3_22 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,2), bias=True, dilation = (1,2))

        self.conv3x1_25 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(5,0), bias=True, dilation = (5,1))
        self.conv1x3_25 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,5), bias=True, dilation = (1,5))

        self.conv3x1_29 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(9,0), bias=True, dilation = (9,1))
        self.conv1x3_29 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,9), bias=True, dilation = (1,9))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, input):
        residual = input
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output2 = self.conv3x1_22(output)
        output2 = F.relu(output2)
        output2 = self.conv1x3_22(output2)
        output2 = self.bn2(output2)
        if (self.dropout.p != 0):
            output2 = self.dropout(output2)

        output5 = self.conv3x1_25(output)
        output5 = F.relu(output5)
        output5 = self.conv1x3_25(output5)
        output5 = self.bn2(output5)
        if (self.dropout.p != 0):
            output5 = self.dropout(output5)

        output9 = self.conv3x1_29(output)
        output9 = F.relu(output9)
        output9 = self.conv1x3_29(output9)
        output9 = self.bn2(output9)
        if (self.dropout.p != 0):
            output9 = self.dropout(output9)

        return F.relu(residual+output2+output5+output9,inplace=True)

		
class ESNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        #-----ESNET---------#
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()
        
        for x in range(0, 3):
           self.layers.append(FCU(16, 3, 0.03, 1))  
        
        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 2):
           self.layers.append(FCU(64, 5, 0.03, 1))  

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 3):   
            self.layers.append(PFCU(chann=128)) 

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(FCU(64, 5, 0, 1))
        self.layers.append(FCU(64, 5, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(FCU(16, 3, 0, 1))
        self.layers.append(FCU(16, 3, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESNet(classes=11).to(device)
    summary(model,(3,360,480))
