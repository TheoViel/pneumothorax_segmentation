import cv2
from params import *
from data.dataset import *

def get_mask_dic(df, return_images=False):
    df['EncodedPixels'].fillna('-1', inplace=True)

    df['Image'] = df['Image_Label'].apply(
        lambda x: '_'.join(x.split('_')[:-1]))
    df['Class'] = df['Image_Label'].apply(lambda x: x.split('_')[-1])

    group_img = df[['Image', 'EncodedPixels']].groupby('Image').agg(list)
    rep_classes = group_img['EncodedPixels'].apply(
        pd.Series).rename(columns=lambda x: str(x))
    rep_classes['ClassNumber'] = group_img['EncodedPixels'].apply(
        lambda x: len([i for i in x if i != "-1"]))

    if return_images:
        return rep_classes.drop('ClassNumber', axis=1).to_dict('index'), rep_classes
    else:
        return rep_classes.drop('ClassNumber', axis=1).to_dict('index')


def lower_res_image_test(img_name, ratio=1):
    img = cv2.imread(TEST_IMG_PATH + img_name)
    resize = albu.Resize(
        TRAIN_SHAPE[0] // ratio, TRAIN_SHAPE[1] // ratio, always_apply=True)
    transformed = resize(image=img)
    return transformed['image']


def lower_res_image(mask_dic, img_name, ratio=2, plot=False):
    img = cv2.imread(TRAIN_IMG_PATH + img_name)
    mask = get_masks(img_name, mask_dic)

    resize = albu.Resize(
        TRAIN_SHAPE[0] // ratio, TRAIN_SHAPE[1] // ratio, always_apply=True)
    transformed = resize(image=img, mask=mask)

    img = transformed['image']
    mask = transformed['mask']

    rles = [mask_to_rle(mask[:, :, i]) for i in range(4)]

    if plot:
        return to_tensor(img), to_tensor(mask)
    return img, rles
