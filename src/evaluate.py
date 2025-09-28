import numpy as np

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    mask = np.abs(y_true) > 1e-12
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / (y_true[mask])))) * 100.0

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-12
    return float(np.mean(np.abs(y_pred - y_true) / denom)) * 100.0