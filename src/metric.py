import torch
import numpy as np


def dice_th(pred, truth, eps=1e-8, threshold=0.5):
    n = truth.shape[0]
    
    pred = (pred.view(n, -1) > threshold).float()
    truth = (truth.view(n, -1) > 0.5).float()

    intersect = (pred + truth == 2).sum(-1).float()
    union = pred.sum(-1) + truth.sum(-1).float()

    return ((2.0 * intersect + eps) / (union + eps)).mean()


def dice_np(pred, truth, eps=1e-8, threshold=0.5):
    n = truth.shape[0]
    
    pred = (pred.reshape((n , -1)) > threshold).astype(int)
    truth = (truth.reshape((n, -1)) > threshold).astype(int)

    intersect = (pred + truth == 2).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    return ((2.0 * intersect + eps) / (union + eps)).mean()


# def eval_predictions(dataset, rles_pred):
#     dice = 0
#     for i in range(len(dataset)):
#         img, _, truth, fault = dataset[i]
#         pred = np.array([rle_to_mask(rle, IMG_SHAPE) for rle in rles_pred[4*i: 4*(i+1)]])
#         assert truth.shape == pred.shape
#         dice += dice_np(np.array([pred]), np.array([truth])) / len(dataset)
#     return dice
