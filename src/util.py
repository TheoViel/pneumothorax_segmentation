import os
import torch
import random
import operator
import numpy as np
import pandas as pd

from params import CP_PATH

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False


def save_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    if verbose:
        print(f'\n -> Saving weights to {os.path.join(cp_folder,filename)}\n')
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(
        model,
        filename,
        verbose=1,
        cp_folder=CP_PATH,
        strict=True):
    if verbose:
        print(
            f'\n -> Loading weights from {os.path.join(cp_folder,filename)}\n')
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    cp_folder,
                    filename),
                map_location='cpu'),
            strict=strict)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def count_parameters(model, all=False):
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
