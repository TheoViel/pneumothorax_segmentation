from torch.utils.model_zoo import load_url
from torchvision.models.resnet import ResNet, BasicBlock


class ResNetEncoder(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4, x3, x2, x1, x0

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)


SETTINGS = {
    "resnet34": {
        "name": "resnet34",
        "encoder": ResNetEncoder,
        "pretrained_settings": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "out_shapes": (512, 256, 128, 64, 64),
        "params": {"block": BasicBlock, "layers": [3, 4, 6, 3],},
    }
}


def get_encoder(settings):
    Encoder = settings["encoder"]
    encoder = Encoder(**settings["params"])
    encoder.out_shapes = settings["out_shapes"]

    if settings["pretrained_settings"] is not None:
        encoder.load_state_dict(
            load_url(settings["pretrained_settings"])
        )

    return encoder
