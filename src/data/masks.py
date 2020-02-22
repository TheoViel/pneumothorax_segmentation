import numpy as np
from metric import *
from params import *
from imports import *


def rle_to_mask(rle, shape):
    if rle == '-1':
        return np.zeros(shape)
    width, height = shape
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T


def mask_to_rle(img):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0
    width, height = img.shape
    img = img.T

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def plot_mask(img, mask):
    if type(mask) == str:
        mask = rle_to_mask(mask, img.shape)
        
    mask = (mask > 0.5).astype(int)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=plt.cm.bone)
    plt.imshow(mask, alpha=0.3, cmap="Reds")
    plt.axis(False)
    plt.show()
