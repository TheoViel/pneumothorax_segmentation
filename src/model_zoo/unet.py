import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import *
from torch.nn.functional import interpolate

from model_zoo.resnet import *
from model_zoo.custom_layers import *


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


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm),
            nn.ReLU(inplace=True), ]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x

        x = interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x


class CenterBlock(DecoderBlock):
    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):
    def __init__(self, encoder_channels, decoder_channels=(64, 64, 64, 64), final_channels=1,
                 use_batchnorm=True, attention_type=None, use_hypercolumns=False, use_skips=True):
        super().__init__()
        self.use_hypercolumns = use_hypercolumns
        self.use_skips = use_skips

        in_channels = self.compute_channels(encoder_channels, decoder_channels, use_skips)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)

        global_conv_size = 64
        if use_hypercolumns:
            global_conv_input_size = np.sum(out_channels)
        else:
            global_conv_input_size = out_channels[-1]

        self.global_layer = nn.Sequential(
            nn.Conv2d(global_conv_input_size, global_conv_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(global_conv_size),

            nn.Conv2d(global_conv_size, global_conv_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(global_conv_size),
        )

        self.final_conv = nn.Conv2d(global_conv_size, final_channels, kernel_size=1)

        self.initialize()

    @staticmethod
    def compute_channels(encoder_channels, decoder_channels, use_skips=False):
        if use_skips:
            channels = [
                encoder_channels[0] + encoder_channels[1],
                encoder_channels[2] + decoder_channels[0],
                encoder_channels[3] + decoder_channels[1],
                encoder_channels[4] + decoder_channels[2],
            ]
        else: 
            channels = [
                encoder_channels[0],
                decoder_channels[0],
                decoder_channels[1],
                decoder_channels[2],
            ]
        return channels
 
    def forward(self, x):
        encoder_head = x[0]

        if self.use_skips:
            skips = x[1:]
        else:
            skips = [None, None, None, None]

        x1 = self.layer1([encoder_head, skips[0]])
        x2 = self.layer2([x1, skips[1]])
        x3 = self.layer3([x2, skips[2]])
        x4 = self.layer4([x3, skips[3]])

        h, w = x4.size()[2:]

        if self.use_hypercolumns:
            x = torch.cat([
                F.upsample_bilinear(x1, size=(h, w)),
                F.upsample_bilinear(x2, size=(h, w)),
                F.upsample_bilinear(x3, size=(h, w)),
                x4
            ], 1)
        else:
            x = x4

        x = self.global_layer(x)
        x = F.upsample_bilinear(x, size=(2 * h, 2 * w))

        return self.final_conv(x)


class SegmentationUnet(Model):
    def __init__(self, encoder_settings, num_classes=1, center_block=None, aux_clf=False, softmax=False,
                 use_bn=True, attention_type=None, use_hypercolumns=True):
        super().__init__()
        self.aux_clf = aux_clf
        self.num_classes = num_classes

        self.encoder = get_encoder(encoder_settings)
        encoder_chanels = list(self.encoder.out_shapes).copy()

        if center_block == "aspp":
            self.center = ASPP(self.encoder.out_shapes[0], self.encoder.out_shapes[1])
            encoder_chanels[0] = self.encoder.out_shapes[1]
        elif center_block == "std":
            self.center = CenterBlock(self.encoder.out_shapes[0], self.encoder.out_shapes[0], use_batchnorm=use_bn)
        else:
            self.center = None

        self.decoder = UnetDecoder(encoder_channels=encoder_chanels, final_channels=num_classes,
                                   use_batchnorm=use_bn, attention_type=attention_type, use_hypercolumns=use_hypercolumns)

        self.logit = nn.Sequential(
            nn.Conv2d(encoder_chanels[0] * 2, 32, kernel_size=1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        head, x3, x2, x1, x0 = self.encoder(x)

        if self.center is not None:
            head = self.center(head)
        
        masks = self.decoder([head, x3, x2, x1, x0]).squeeze(1)

        if self.aux_clf:
            x = adaptive_concat_pool2d(head)
            return masks, self.logit(x).view(-1)

        return masks, 0


if __name__ == '__main__':
    print('Building Unet with Resnet34 backbone...')
    _ = SegmentationUnet(SETTINGS['resnet34'], num_classes=4, center_block="aspp", aux_clf=True).to(DEVICE)