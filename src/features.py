import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

try:
    import pandas_ta as ta
except ImportError:
    print("Warning: pandas_ta not available. Some features will be disabled.")
    ta = None

from utils import rolling_beta_corr, standardize_calendar

def _load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("prices_daily.csv missing date/time column")
    df = df.sort_values("date").reset_index(drop=True)
    return df

def _pct_zscore(s: pd.Series, window: int=20) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / (sd + 1e-12)

def _rolling_slope(y: pd.Series, window: int=10) -> pd.Series:
    idx = np.arange(window)
    def slope(arr):
        x = idx
        y = arr
        x = x - x.mean()
        y = y - y.mean()
        denom = (x**2).sum() + 1e-12
        return (x*y).sum() / denom
    return y.rolling(window).apply(slope, raw=True)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ret"] = np.log(out["close"]).diff()
    out["ret1"] = out["log_ret"]
    out["ret5"] = np.log(out["close"]).diff(5)
    out["ret10"] = np.log(out["close"]).diff(10)

    if ta is not None:
        for k in [5,10,20,50]:
            out[f"sma{k}"] = ta.sma(out["close"], length=k)
            out[f"sma{k}_ratio"] = out["close"]/out[f"sma{k}"] - 1.0
    out["ema12"] = ta.ema(out["close"], length=12)
    out["ema26"] = ta.ema(out["close"], length=26)
    out["ema_ratio"] = out["ema12"]/out["ema26"] - 1.0
    out["slope_sma20"] = _rolling_slope(out["sma20"], 20)

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd"] = macd["MACD_12_26_9"]
        out["macd_signal"] = macd["MACDs_12_26_9"]
        out["macd_hist"] = macd["MACDh_12_26_9"]

    out["rsi14"] = ta.rsi(out["close"], length=14)
    stoch = ta.stoch(out["high"], out["low"], out["close"], k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        out["stoch_k"] = stoch["STOCHk_14_3_3"]
        out["stoch_d"] = stoch["STOCHd_14_3_3"]
    out["willr14"] = ta.willr(out["high"], out["low"], out["close"], length=14)
    out["roc1"] = ta.roc(out["close"], length=1)
    out["roc5"] = ta.roc(out["close"], length=5)
    out["roc10"] = ta.roc(out["close"], length=10)

    out["atr14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["atr14_pct"] = out["atr14"] / (out["close"].abs() + 1e-12)
    bb = ta.bbands(out["close"], length=20, std=2.0)
    if bb is not None and not bb.empty:
        out["bb_bw"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / (bb["BBM_20_2.0"] + 1e-12)
        out["bb_pctb"] = (out["close"] - bb["BBL_20_2.0"]) / ((bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) + 1e-12)
    out["hv20"] = out["log_ret"].rolling(20).std()

    out["range_hl_pct"] = (out["high"] - out["low"]) / (out["close"].abs() + 1e-12)
    out["range_co_pct"] = (out["close"] - out["open"]).abs() / (out["close"].abs() + 1e-12)

    out["volume_z20"] = _pct_zscore(out["volume"].astype(float), 20)
    obv = ta.obv(out["close"], out["volume"])
    out["obv"] = obv.astype(float) if obv is not None else 0.0
    out["obv_slope10"] = _rolling_slope(out["obv"], 10)
    mfi_result = ta.mfi(out["high"], out["low"], out["close"], out["volume"], length=14)
    out["mfi14"] = mfi_result.astype(float) if mfi_result is not None else 0.0
    out["dollar_vol"] = (out["close"] * out["volume"]).astype(float)

    out["hhv252"] = out["close"].rolling(252).max()
    out["llv252"] = out["close"].rolling(252).min()
    out["pct_from_52w_high"] = out["close"] / (out["hhv252"] + 1e-12) - 1.0
    out["pct_from_52w_low"]  = out["close"] / (out["llv252"] + 1e-12) - 1.0
    out["rolling_max_60"] = out["close"].rolling(60).max()
    out["drawdown_60"] = out["close"] / (out["rolling_max_60"] + 1e-12) - 1.0

    out["close_lag1"] = out["close"].shift(1)
    out["open_lag0"] = out["open"]
    out["overnight_ret"] = np.log((out["open_lag0"] + 1e-12) / (out["close_lag1"] + 1e-12))
    out["intraday_ret"] = np.log((out["close"] + 1e-12) / (out["open_lag0"] + 1e-12))

    return out

def add_market_features(stock_df: pd.DataFrame, index_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = stock_df.copy()
    if index_df is None:
        return out
    idx = index_df.rename(columns={c: c.lower() for c in index_df.columns})
    if "time" in idx.columns:
        idx["date"] = pd.to_datetime(idx["time"])
    else:
        idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["idx_log_ret"] = np.log(idx["close"]).diff()
    out = out.merge(idx[["date","close","idx_log_ret"]].rename(columns={"close":"index_close"}), on="date", how="left")
    out["rel_ret1"] = out["log_ret"] - out["idx_log_ret"]
    beta60, corr60 = rolling_beta_corr(out["log_ret"], out["idx_log_ret"], window=60)
    out["beta60"] = beta60
    out["corr60"] = corr60
    return out

def finalize_features(df: pd.DataFrame, use_calendar: bool=True, use_regime: bool=True) -> pd.DataFrame:
    out = df.copy()
    if use_calendar:
        out = standardize_calendar(out)
    if use_regime:
        hv20 = out["hv20"]
        out["vol_regime_high"] = (hv20 > hv20.rolling(252).quantile(0.8)).astype(int)
        from pandas_ta import adx
        adx_df = adx(out["high"], out["low"], out["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            out["adx14"] = adx_df["ADX_14"].astype(float)
            out["trend_regime"] = (out["adx14"] > 25).astype(int)
        else:
            out["trend_regime"] = 0
    
    # Đảm bảo tất cả columns là numeric
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce').astype(float)
    
    return out

def build_supervised(df: pd.DataFrame, predict_price: bool = False) -> pd.DataFrame:
    """
    Build supervised learning targets

    Args:
        df: DataFrame with price data
        predict_price: If True, predict next day price. If False, predict log return.
    """
    out = df.copy()

    if predict_price:
        # Predict next day price (for baseline comparison)
        out["y"] = out["close"].shift(-1)
    else:
        # Predict log return (default for DL models)
        out["y"] = np.log(out["close"]).shift(-1) - np.log(out["close"])

    out = out.iloc[:-1].reset_index(drop=True)
    return out

def windowize(arr: np.ndarray, y: np.ndarray, lookback: int):
    xs, ys = [], []
    for i in range(len(arr) - lookback + 1):
        j = i + lookback - 1
        if j >= len(y):
            break
        xs.append(arr[i:i+lookback, :])
        ys.append(y[j])
    return np.asarray(xs), np.asarray(ys)