import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    diff = np.where(np.isfinite(diff), diff, 0.0)
    return float(np.sqrt(np.mean(diff ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = np.abs(y_pred - y_true)
    diff = np.where(np.isfinite(diff), diff, 0.0)
    return float(np.mean(diff))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & (np.abs(y_true) > 1e-8) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return 0.0
    val = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
    val = np.where(np.isfinite(val), val, 0.0)
    return float(np.mean(val) * 100.0)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(np.isfinite(denom), denom, 0.0)
    denom = denom + 1e-8
    num = np.abs(y_pred - y_true)
    num = np.where(np.isfinite(num), num, 0.0)
    val = num / denom
    val = np.where(np.isfinite(val), val, 0.0)
    return float(np.mean(val) * 100.0)