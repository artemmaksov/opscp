import modin.pandas as pd
import numpy as np
import torch

def mrrmse_pd(y_pred: pd.DataFrame, y_true: pd.DataFrame):
    return ((y_pred - y_true)**2).mean(axis=1).apply(np.sqrt).mean()

def mrrmse_np(y_pred, y_true):
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()

def mrrmse_torch(y_pred: torch.Tensor, y_true: torch.Tensor):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=1)).mean()
