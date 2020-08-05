from model.SQNet import SQNet
from model.LinkNet import LinkNet
from model.SegNet import SegNet
from model.UNet import UNet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.CGNet import CGNet
from model.EDANet import EDANet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.LEDNet import LEDNet
from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.ContextNet import ContextNet
from model.FastSCNN import FastSCNN
from model.DABNet import DABNet
from model.FSSNet import FSSNet
from model.FPENet import FPENet
# from model.PSPNet.pspnet import PSPNet



def build_model(model_name, num_classes):
    # small model
    if model_name == 'SQNet':
        return SQNet(classes=num_classes)
    elif model_name == 'LinkNet':
        return LinkNet(classes=num_classes)
    elif model_name == 'SegNet':
        return SegNet(classes=num_classes)
    elif model_name == 'ENet':
        return ENet(classes=num_classes)
    elif model_name == 'ERFNet':
        return ERFNet(classes=num_classes)
    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)
    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)
    elif model_name == 'ESNet':
        return ESNet(classes=num_classes)
    elif model_name == 'ESPNet':
        return ESPNet(classes=num_classes)
    elif model_name == 'LEDNet':
        return LEDNet(classes=num_classes)
    elif model_name == 'ESPNet_v2':
        return EESPNet_Seg(classes=num_classes)
    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)
    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes)
    elif model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'FSSNet':
        return FSSNet(classes=num_classes)
    elif model_name == 'FPENet':
        return FPENet(classes=num_classes)
    # large model
    elif model_name == 'UNet':
        return UNet(classes=num_classes)
    # elif model_name == 'PSPNet50':
    #     return PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=num_classes, zoom_factor=1, use_ppm=True, pretrained=True)
    # elif model_name == 'PSANet50':
    #     return PSANet(layers=50, dropout=0.1, classes=num_classes, zoom_factor=8, use_psa=True, psa_type=2, compact=compact,
    #                shrink_factor=shrink_factor, mask_h=mask_h, mask_w=mask_w, psa_softmax=True, pretrained=True)
    # elif