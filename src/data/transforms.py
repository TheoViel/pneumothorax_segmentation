from util import *
from params import *
from imports import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_transfos(flip_only=False, **kwargs):
    transforms = albu.Compose([
        # albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=45, p=0.5),
        albu.HorizontalFlip(p=0.5),
        # albu.VerticalFlip(p=0.5),
    ])


    return transforms