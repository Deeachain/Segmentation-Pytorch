import sys
import argparse
import torch
from utils.flops_counter.ptflops import get_model_complexity_info

from model.UNet import UNet
from model.FCN8s import FCN
from model.ENet import ENet
from model.SegNet import SegNet
from model.ERFNet import ERFNet
from model.ESPNet import ESPNet
from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.DABNet import DABNet
from model.BiSeNet import BiSeNet
from model.BiSeNetV2 import BiSeNetV2
from model.PSPNet.pspnet import PSPNet
# from model.PSPNet.psanet import PSANet
from model.DeeplabV3Plus import Deeplabv3plus_res50
from model.DualGCNNet import DualSeg_res50
from model.MyNet import MyNet
from model.MyNet_trans import MyNet_trans
from model.NFSNet import NFSNet

models = {
    'ENet': ENet,
    'FCN': FCN,
    'UNet': UNet,
    'BiSeNet': BiSeNet,
    'BiSeNetV2': BiSeNetV2,
    'PSPNet': PSPNet,
    'DeeplabV3Plus': Deeplabv3plus_res50,
    'DualGCNNet': DualSeg_res50,
    'MyNet': MyNet,
    'MyNet_trans': MyNet_trans,
    'NFSNet': NFSNet
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0, help='Device to store the model.')
    parser.add_argument('--model', choices=list(models.keys()), type=str, default='NFSNet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    net = models[args.model](num_classes=3)

    flops, params = get_model_complexity_info(net, (3, 512, 512), as_strings=True, print_per_layer_stat=True, ost=ost)

    print('Flops: ' + flops)
    print('Params: ' + params)
