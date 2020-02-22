from util import *
from params import *
from imports import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

   
def crop_img_mask(img, mask, crop_x, crop_y):
    h, w, c = img.shape

    x_img = np.random.randint(0, h - crop_x)
    y_img = np.random.randint(0, w - crop_y)
    x_mask, y_mask = x_img // 2, y_img // 2
    
    return (img[x_img : x_img + crop_x, y_img : y_img + crop_y, :],
            mask[x_mask : x_mask + crop_x // 2 , y_mask : y_mask + crop_y // 2, :])


def get_transfos(flip_only=False, **kwargs):
    transforms = albu.Compose([
        # albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=45, p=0.5),
        albu.HorizontalFlip(p=0.5),
        # albu.VerticalFlip(p=0.5),
    ])


    return transforms