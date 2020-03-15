import os
import torch
import numpy as np
from datetime import date


seed = 2019

DATA_PATH = '../input/siim/'

TEST_IMG_PATH = DATA_PATH + 'dicom-images-test/'
TRAIN_IMG_PATH = DATA_PATH + 'dicom-images-train/'  # Original images

TRAIN_IMG_PATH_4 = DATA_PATH  + 'train_images_4/' # 1/4 rez Image


CP_PATH = f'../checkpoints/{date.today()}/'
if not os.path.exists(CP_PATH):
    os.mkdir(CP_PATH) 

IMG_SHAPE = (1024, 1024)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4
VAL_BS = 4 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")