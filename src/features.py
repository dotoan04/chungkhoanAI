import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

try:
    import pandas_ta as ta
except ImportError:
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
            sma_result = ta.sma(out["close"], length=k)
            if sma_result is not None:
                out[f"sma{k}"] = sma_result
                out[f"sma{k}_ratio"] = out["close"]/out[f"sma{k}"] - 1.0
        ema12 = ta.ema(out["close"], length=12)
        ema26 = ta.ema(out["close"], length=26)
        if ema12 is not None and ema26 is not None:
            out["ema12"] = ema12
            out["ema26"] = ema26
            out["ema_ratio"] = out["ema12"]/out["ema26"] - 1.0
        if "sma20" in out.columns:
            out["slope_sma20"] = _rolling_slope(out["sma20"], 20)
    else:
        # Create dummy values when pandas_ta is not available
        for k in [5,10,20,50]:
            out[f"sma{k}"] = out["close"].rolling(k).mean()
            out[f"sma{k}_ratio"] = out["close"]/out[f"sma{k}"] - 1.0
        out["ema12"] = out["close"].ewm(span=12).mean()
        out["ema26"] = out["close"].ewm(span=26).mean()
        out["ema_ratio"] = out["ema12"]/out["ema26"] - 1.0
        out["slope_sma20"] = _rolling_slope(out["sma20"], 20)

    if ta is not None:
        macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            out["macd"] = macd["MACD_12_26_9"]
            out["macd_signal"] = macd["MACDs_12_26_9"]
            out["macd_hist"] = macd["MACDh_12_26_9"]

        rsi14 = ta.rsi(out["close"], length=14)
        if rsi14 is not None:
            out["rsi14"] = rsi14

        stoch = ta.stoch(out["high"], out["low"], out["close"], k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            out["stoch_k"] = stoch["STOCHk_14_3_3"]
            out["stoch_d"] = stoch["STOCHd_14_3_3"]

        willr14 = ta.willr(out["high"], out["low"], out["close"], length=14)
        if willr14 is not None:
            out["willr14"] = willr14

        roc1 = ta.roc(out["close"], length=1)
        if roc1 is not None:
            out["roc1"] = roc1
        roc5 = ta.roc(out["close"], length=5)
        if roc5 is not None:
            out["roc5"] = roc5
        roc10 = ta.roc(out["close"], length=10)
        if roc10 is not None:
            out["roc10"] = roc10

        atr14 = ta.atr(out["high"], out["low"], out["close"], length=14)
        if atr14 is not None:
            out["atr14"] = atr14
    else:
        # Create dummy values when pandas_ta is not available
        # For indicators we can't easily replicate, set to 0 or neutral values
        out["macd"] = 0.0
        out["macd_signal"] = 0.0
        out["macd_hist"] = 0.0
        out["rsi14"] = 50.0  # Neutral RSI value
        out["stoch_k"] = 50.0
        out["stoch_d"] = 50.0
        out["willr14"] = -50.0  # Neutral Williams %R value
        out["roc1"] = 0.0
        out["roc5"] = 0.0
        out["roc10"] = 0.0
        out["atr14"] = out["close"] * 0.01  # Approximate ATR as 1% of price
    if "atr14" in out.columns:
        out["atr14_pct"] = out["atr14"] / (out["close"].abs() + 1e-12)

    if ta is not None:
        try:
            bb = ta.bbands(out["close"], length=20, std=2.0)
        except Exception as e:
            print(f"[ERROR] Failed to calculate Bollinger Bands: {e}")
            bb = None
    else:
        # Create simple Bollinger Bands approximation using rolling mean and std
        rolling_mean = out["close"].rolling(20).mean()
        rolling_std = out["close"].rolling(20).std()
        bb_upper = rolling_mean + 2 * rolling_std
        bb_lower = rolling_mean - 2 * rolling_std
        bb_mean = rolling_mean

        # Create a DataFrame-like object for compatibility
        class SimpleBB:
            def __init__(self, upper, lower, mean):
                self.BBU_20_2_0 = upper
                self.BBL_20_2_0 = lower
                self.BBM_20_2_0 = mean
                self.empty = False

        bb = SimpleBB(bb_upper, bb_lower, bb_mean)

    if bb is not None and not getattr(bb, 'empty', True):
        # Use simple object attributes for fallback calculation
        bbu_vals = bb.BBU_20_2_0
        bbl_vals = bb.BBL_20_2_0
        bbm_vals = bb.BBM_20_2_0

        out["bb_bw"] = (bbu_vals - bbl_vals) / (bbm_vals + 1e-12)
        out["bb_pctb"] = (out["close"] - bbl_vals) / ((bbu_vals - bbl_vals) + 1e-12)
    else:
        out["bb_bw"] = 0.0
        out["bb_pctb"] = 0.5

    out["hv20"] = out["log_ret"].rolling(20).std()

    out["range_hl_pct"] = (out["high"] - out["low"]) / (out["close"].abs() + 1e-12)
    out["range_co_pct"] = (out["close"] - out["open"]).abs() / (out["close"].abs() + 1e-12)

    out["volume_z20"] = _pct_zscore(out["volume"].astype(float), 20)

    if ta is not None:
        obv = ta.obv(out["close"], out["volume"])
        out["obv"] = obv.astype(float) if obv is not None else 0.0
        mfi_result = ta.mfi(out["high"], out["low"], out["close"], out["volume"], length=14)
        out["mfi14"] = mfi_result.astype(float) if mfi_result is not None else 0.0
    else:
        # Create simple approximations when pandas_ta is not available
        out["obv"] = (out["close"].diff() * out["volume"]).fillna(0).cumsum()
        # MFI approximation (simplified)
        out["mfi14"] = 50.0  # Neutral MFI value
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

        if ta is not None:
            from pandas_ta import adx
            adx_df = adx(out["high"], out["low"], out["close"], length=14)
            if adx_df is not None and not adx_df.empty:
                out["adx14"] = adx_df["ADX_14"].astype(float)
                out["trend_regime"] = (out["adx14"] > 25).astype(int)
            else:
                out["trend_regime"] = 0
        else:
            # Simple trend regime approximation when pandas_ta is not available
            # Use price momentum as a proxy
            out["adx14"] = 25.0  # Neutral ADX value
            out["trend_regime"] = 0  # Default to no trend
    
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