from util import *
from params import *
from imports import *
from data.masks import *
from data.transforms import *

from fastai.vision import *

import pydicom


def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.
    Taken from https://www.kaggle.com/ekhtiar/finding-pneumo-part-1-eda-and-unet/data
    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.

    Returns:
        dict: contains metadata of relevant fields.
    """

    data = {}

    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID

    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId'] ==
                                      dicom_data.SOPInstanceUID]['EncodedPixels'].values

        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != '-1':
                pneumothorax = True

        # get meaningful information (for train set)
        try:
            data['rle'] = encoded_pixels_list[0]
        except:
            data['rle'] = '-1'
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)

    return data


class PneumoDataset(Dataset):
    def __init__(self, df, transforms=None, ratio=1):
        super().__init__()

        self.path = DATA_PATH + f'train_images_{ratio}/'
        self.img_shape = (IMG_SHAPE[0] // ratio, IMG_SHAPE[1] // ratio)

        self.img_names = df['id'].values
        self.y = df['has_pneumothorax'].values.astype(int)
        self.rles = df[f'rle_{ratio}'].fillna('-1').values

        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = cv2.imread(self.path + self.img_names[idx] + '.png')
        img = img / 255

        mask = rle_to_mask(self.rles[idx], self.img_shape).astype(float) / 255

#         if np.array(self.crop_size).any():
#             img, mask = crop_img_mask(img, mask, self.crop_size[0], self.crop_size[1])

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = (img - MEAN) / STD
        return torch.tensor(to_tensor(img)), torch.tensor(mask, dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)