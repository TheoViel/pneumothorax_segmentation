from imports import *


class HyperColumn(Module):
    def __init__(self, input_channels: list, output_channels: list, im_size: int, kernel_size=1):
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer=None):
        bs, ch, *image_size = last_layer.shape
        up = nn.Upsample(image_size, mode='bilinear')
        hcs = [up(c(x)) for c, x in zip(self.convs, xs)]
        if last_layer is not None:
            hcs.append(last_layer)
        return torch.cat(hcs, dim=1)