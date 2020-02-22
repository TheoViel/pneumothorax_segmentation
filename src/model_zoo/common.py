from imports import *
from model_zoo.resnet import *
from model_zoo.senet import *


SETTINGS = {'resnet34':
            {'name': 'resnet34',
             'encoder': ResNetEncoder,
             'pretrained_settings': pretrained_settings['resnet34']['imagenet'],
             'out_shapes': (512, 256, 128, 64, 64),
             'params': {'block': BasicBlock, 'layers': [3, 4, 6, 3],}
            },
            'se_resnext50_32x4d': {
                'encoder': SENetEncoder,
                'pretrained_settings': pretrained_settings_senet['se_resnext50_32x4d']['imagenet'],
                'out_shapes': (2048, 1024, 512, 256, 64),
                'params': {
                    'block': SEResNeXtBottleneck,
                    'layers': [3, 4, 6, 3],
                    'downsample_kernel_size': 1,
                    'downsample_padding': 0,
                    'dropout_p': None,
                    'groups': 32,
                    'inplanes': 64,
                    'input_3x3': False,
                    'num_classes': 1000,
                    'reduction': 16}
            }
            }


def get_encoder(settings):
    Encoder = settings['encoder']
    encoder = Encoder(**settings['params'])
    encoder.out_shapes = settings['out_shapes']

    if settings['pretrained_settings'] is not None:
        encoder.load_state_dict(model_zoo.load_url(settings['pretrained_settings']['url']))

    return encoder


def adaptive_concat_pool2d(x, sz=1):
    out1 = F.adaptive_avg_pool2d(x, sz)
    out2 = F.adaptive_max_pool2d(x, sz)
    return torch.cat([out1, out2], 1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
