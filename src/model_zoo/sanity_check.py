from params import *
from model_zoo.unet import *
from model_zoo.common import SETTINGS

if __name__ == '__main__':
    print('Building Unet with Resnet34 backbone...')
    _ = SegmentationUnet(SETTINGS['resnet34'], num_classes=4, center_block="aspp", aux_clf=True).to(DEVICE)
    print('Building Unet with SeResNext50 backbone...')
    _ = SegmentationUnet(SETTINGS['se_resnext50_32x4d'], num_classes=4, center_block="aspp", aux_clf=True).to(DEVICE)