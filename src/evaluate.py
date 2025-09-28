import numpy as np
import json
from pathlib import Path

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

# Directional accuracy for a given horizon
def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return 0.0
    sign_true = (y_true[mask] > 0).astype(int)
    sign_pred = (y_pred[mask] > 0).astype(int)
    return float((sign_true == sign_pred).mean())

# Information Coefficient (Pearson correlation)
def information_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return 0.0
    if np.std(y_true[mask]) < 1e-12 or np.std(y_pred[mask]) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0,1])

# Sharpe-like metric on predictions
def sharpe_like(y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = y_pred[np.isfinite(y_pred)]
    if y_pred.size == 0:
        return 0.0
    std = np.std(y_pred)
    if std < 1e-12:
        return 0.0
    return float(np.mean(y_pred) / std)

def log_metrics_json(path: Path, metrics_obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, ensure_ascii=False, indent=2)