# tests/test_loss.py
import numpy as np
import pandas as pd
import torch
import pytest

from src.utils.loss import mrrmse_pd, mrrmse_np, mrrmse_torch

@pytest.fixture
def data():
    y_true = np.array([[1, 2, 3], [4, 5, 6]])
    y_pred_same = np.array([[1, 2, 3], [4, 5, 6]])
    y_pred_diff = np.array([[2, 3, 4], [4, 5, 6]])
    return y_true, y_pred_same, y_pred_diff

def test_mrrmse_pd(data):
    y_true, y_pred_same, y_pred_diff = data
    assert mrrmse_pd(pd.DataFrame(y_pred_same), pd.DataFrame(y_true)) == 0
    assert mrrmse_pd(pd.DataFrame(y_pred_diff), pd.DataFrame(y_true)) == 0.5

def test_mrrmse_np(data):
    y_true, y_pred_same, y_pred_diff = data
    assert mrrmse_np(y_pred_same, y_true) == 0
    assert mrrmse_np(y_pred_diff, y_true) == 0.5

def test_mrrmse_torch(data):
    y_true, y_pred_same, y_pred_diff = data
    assert mrrmse_torch(torch.tensor(y_pred_same, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32)) == 0
    assert mrrmse_torch(torch.tensor(y_pred_diff, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32)) == 0.5