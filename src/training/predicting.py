import cv2
import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm

from params import *


def save_mask(mask, name='mask.png', img_folder="../masks/"):
    mask = (mask * 255).astype(np.uint8).transpose(1, 2, 0)
    mask = cv2.resize(mask, (525, 350))
    cv2.imwrite(img_folder + name, mask)


def predict_and_save(dataset, model, img_names, tta=False, img_folder="../masks/"):
    model.eval()
    preds = np.array([[], [], [], []]).T
    preds_max = np.array([[], [], [], []]).T
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for i, (x, _, _) in enumerate(tqdm(loader)):
            masks, prob = model(x.to(DEVICE))
            masks = torch.sigmoid(masks.detach()).cpu().numpy()
            probs_max = masks.max(-1).max(-1)
            probs = torch.sigmoid(prob.detach()).cpu().numpy()

            if tta:
                flips = [[-1], [-2], [-2, -1]]
                for f in flips:
                    mask, prob = model(torch.flip(x.to(DEVICE), f))
                    mask = torch.sigmoid(torch.flip(mask, f).detach()).cpu().numpy()
                    masks += mask
                    probs_max += mask.max(-1).max(-1)
                    probs += torch.sigmoid(prob.detach()).cpu().numpy()
                    
                masks /= len(flips) + 1
                probs /= len(flips) + 1
                probs_max /= len(flips) + 1
                
            preds = np.concatenate([preds, probs])
            preds_max = np.concatenate([preds_max, probs_max])
            save_mask(masks[0], name=img_names[i][:-4] + '.png', img_folder=img_folder)
            
    return preds, preds_max