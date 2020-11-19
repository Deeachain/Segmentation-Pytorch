from model.UNet import UNet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.ESPNet import ESPNet
from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.DABNet import DABNet
from model.BiSeNetV2 import BiSeNetV2
from model.PSPNet.pspnet import PSPNet
# from model.PSPNet.psanet import PSANet
from model.DeeplabV3Plus.DeeplabV3Plus import Deeplabv3plus
from model.DeeplabV3Plus.config import cfg
from model.DualGCNNet import DualSeg_res50




def build_model(model_name, num_classes):
    # small model
    if model_name == 'ENet':
        return ENet(classes=num_classes)
    elif model_name == 'ERFNet':
        return ERFNet(classes=num_classes)
    elif model_name == 'ESPNet':
        return ESPNet(classes=num_classes)
    elif model_name == 'ESPNet_v2':
        return EESPNet_Seg(classes=num_classes)
    elif model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'BiSeNetV2':
        return BiSeNetV2(n_classes=num_classes)

    # large model
    elif model_name == 'UNet':
        return UNet(classes=num_classes)
    elif model_name == 'PSPNet50':
        return PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=num_classes, zoom_factor=8, use_ppm=True, pretrained=True)
    # elif model_name == 'PSANet50':
    #     return PSANet(layers=50, dropout=0.1, classes=num_classes, zoom_factor=8, use_psa=True, psa_type=2, compact=compact,
    #                shrink_factor=shrink_factor, mask_h=mask_h, mask_w=mask_w, psa_softmax=True, pretrained=True)
    elif model_name == 'Deeplabv3plus':
        return Deeplabv3plus(cfg, num_classes=num_classes)

    # gcn
    elif model_name == 'DualSeg_res50':
        return DualSeg_res50(num_classes=num_classes)