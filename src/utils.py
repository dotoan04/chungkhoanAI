import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

from sklearn.preprocessing import RobustScaler

def seed_everything(seed: int = 42):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def rolling_beta_corr(x: pd.Series, y: pd.Series, window: int = 60) -> Tuple[pd.Series, pd.Series]:
    cov = x.rolling(window).cov(y)
    var = y.rolling(window).var()
    beta = cov / (var + 1e-12)
    corr = x.rolling(window).corr(y)
    return beta, corr

def walk_forward_splits(n: int, val_size: int, test_size: int, min_train: int):
    # Generate expanding-window splits for indices [0, n). Yields (train, val, test).
    # This function returns one sequence of successive splits.
    start = 0
    # place first train end at min_train, then slide by test_size
    for t_end in range(min_train, n - (val_size + test_size) + 1, test_size):
        tr = slice(0, t_end)
        va = slice(t_end, t_end + val_size)
        te = slice(t_end + val_size, t_end + val_size + test_size)
        yield tr, va, te

def standardize_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["time"] if "time" in df.columns else df["date"])
    df["dow"] = df["date"].dt.weekday  # 0=Mon
    for d in range(7):
        df[f"dow_{d}"] = (df["dow"] == d).astype(int)
    m = df["date"].dt.month.astype(float)
    df["month_sin"] = np.sin(2*np.pi*m/12.0)
    df["month_cos"] = np.cos(2*np.pi*m/12.0)
    return df

@dataclass
class ScalerBundle:
    x_scaler: RobustScaler
    y_scaler: RobustScaler

def fit_transform_scalers(X_tr: np.ndarray, y_tr: np.ndarray):
    xs = RobustScaler()
    ys = RobustScaler()
    X_tr2 = xs.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    y_tr2 = ys.fit_transform(y_tr.reshape(-1, 1)).reshape(-1)
    return ScalerBundle(xs, ys), X_tr2, y_tr2

def apply_scalers(bundle: ScalerBundle, X: np.ndarray, y: np.ndarray):
    X2 = bundle.x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y2 = bundle.y_scaler.transform(y.reshape(-1, 1)).reshape(-1)
    return X2, y2

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)