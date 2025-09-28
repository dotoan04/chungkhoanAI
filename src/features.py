import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pandas_ta not available ({e}). Some features will be disabled.")
    print("Install with: pip install pandas-ta")
    ta = None
    PANDAS_TA_AVAILABLE = False

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

    if PANDAS_TA_AVAILABLE and ta is not None:
        # Technical indicators using pandas-ta
        for k in [5,10,20,50]:
            out[f"sma{k}"] = ta.sma(out["close"], length=k)
            out[f"sma{k}_ratio"] = out["close"]/out[f"sma{k}"] - 1.0
        out["ema12"] = ta.ema(out["close"], length=12)
        out["ema26"] = ta.ema(out["close"], length=26)
        out["ema_ratio"] = out["ema12"]/out["ema26"] - 1.0
        out["slope_sma20"] = _rolling_slope(out["sma20"], 20)
    else:
        # Fallback: create basic indicators manually
        print("Using manual indicators (pandas-ta not available)")
        for k in [5,10,20,50]:
            out[f"sma{k}"] = out["close"].rolling(k).mean()
            out[f"sma{k}_ratio"] = out["close"]/out[f"sma{k}"] - 1.0

        # Simple EMA approximation using SMA
        out["ema12"] = out["close"].rolling(12).mean()
        out["ema26"] = out["close"].rolling(26).mean()
        out["ema_ratio"] = out["ema12"]/out["ema26"] - 1.0

        # Slope of SMA20
        if f"sma20" in out.columns:
            out["slope_sma20"] = _rolling_slope(out["sma20"], 20)
        else:
            out["slope_sma20"] = 0.0

    if PANDAS_TA_AVAILABLE and ta is not None:
        # MACD with error handling
        try:
            macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                # Find MACD columns dynamically
                macd_cols = [col for col in macd.columns if 'MACD' in col]
                if len(macd_cols) >= 3:
                    # Get MACD columns
                    macd_col = next(col for col in macd_cols if 'MACD' in col and not any(x in col for x in ['s', 'h']))
                    macd_signal_col = next(col for col in macd_cols if 'MACDs' in col or 'signal' in col.lower())
                    macd_hist_col = next(col for col in macd_cols if 'MACDh' in col or 'hist' in col.lower())

                    out["macd"] = macd[macd_col]
                    out["macd_signal"] = macd[macd_signal_col]
                    out["macd_hist"] = macd[macd_hist_col]
                else:
                    # Fallback: skip MACD
                    print("[WARN] MACD columns not found, skipping MACD indicators")
        except Exception as e:
            print(f"[WARN] MACD calculation failed: {e}")

        out["rsi14"] = ta.rsi(out["close"], length=14)

        # Stochastic Oscillator with error handling
        try:
            stoch = ta.stoch(out["high"], out["low"], out["close"], k=14, d=3, smooth_k=3)
            if stoch is not None and not stoch.empty:
                # Find Stochastic columns dynamically
                stoch_cols = [col for col in stoch.columns if 'STOCH' in col]
                if len(stoch_cols) >= 2:
                    # Get Stochastic columns
                    stoch_k_col = next(col for col in stoch_cols if 'k' in col.lower())
                    stoch_d_col = next(col for col in stoch_cols if 'd' in col.lower())

                    out["stoch_k"] = stoch[stoch_k_col]
                    out["stoch_d"] = stoch[stoch_d_col]
                else:
                    print("[WARN] Stochastic columns not found, skipping Stochastic indicators")
        except Exception as e:
            print(f"[WARN] Stochastic calculation failed: {e}")

        out["willr14"] = ta.willr(out["high"], out["low"], out["close"], length=14)
        out["roc1"] = ta.roc(out["close"], length=1)
        out["roc5"] = ta.roc(out["close"], length=5)
        out["roc10"] = ta.roc(out["close"], length=10)

        out["atr14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
        out["atr14_pct"] = out["atr14"] / (out["close"].abs() + 1e-12)
    else:
        # Fallback indicators when pandas-ta not available
        print("Using manual indicators for remaining features")

        # Simple RSI approximation
        out["rsi14"] = 50.0  # Neutral RSI

        # Simple stochastic approximation
        out["stoch_k"] = 50.0
        out["stoch_d"] = 50.0

        out["willr14"] = 0.0  # Williams %R
        out["roc1"] = out["close"].pct_change(1) * 100
        out["roc5"] = out["close"].pct_change(5) * 100
        out["roc10"] = out["close"].pct_change(10) * 100

        # Simple ATR approximation
        out["atr14"] = out["close"].rolling(14).std()
        out["atr14_pct"] = out["atr14"] / (out["close"].abs() + 1e-12)

    if PANDAS_TA_AVAILABLE and ta is not None:
        # Bollinger Bands with error handling for different pandas-ta versions
        try:
            bb = ta.bbands(out["close"], length=20, std=2.0)
            if bb is not None and not bb.empty:
                # Find Bollinger Bands columns dynamically
                bb_cols = [col for col in bb.columns if 'BBU' in col or 'BBL' in col or 'BBM' in col]
                if len(bb_cols) >= 3:
                    # Get the first matching columns (should be BBU, BBL, BBM)
                    bbu_col = next(col for col in bb_cols if 'BBU' in col)
                    bbl_col = next(col for col in bb_cols if 'BBL' in col)
                    bbm_col = next(col for col in bb_cols if 'BBM' in col)

                    out["bb_bw"] = (bb[bbu_col] - bb[bbl_col]) / (bb[bbm_col] + 1e-12)
                    out["bb_pctb"] = (out["close"] - bb[bbl_col]) / ((bb[bbu_col] - bb[bbl_col]) + 1e-12)
                else:
                    # Fallback: create simple Bollinger Bands manually
                    rolling_mean = out["close"].rolling(20).mean()
                    rolling_std = out["close"].rolling(20).std()
                    out["bb_bw"] = (2 * rolling_std) / (rolling_mean + 1e-12)
                    out["bb_pctb"] = (out["close"] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std + 1e-12)
            else:
                # Create simple Bollinger Bands manually if pandas-ta fails
                rolling_mean = out["close"].rolling(20).mean()
                rolling_std = out["close"].rolling(20).std()
                out["bb_bw"] = (2 * rolling_std) / (rolling_mean + 1e-12)
                out["bb_pctb"] = (out["close"] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std + 1e-12)
        except Exception as e:
            print(f"[WARN] Bollinger Bands calculation failed: {e}")
            # Create simple Bollinger Bands manually as fallback
            rolling_mean = out["close"].rolling(20).mean()
            rolling_std = out["close"].rolling(20).std()
            out["bb_bw"] = (2 * rolling_std) / (rolling_mean + 1e-12)
            out["bb_pctb"] = (out["close"] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std + 1e-12)
    else:
        # Manual Bollinger Bands when pandas-ta not available
        rolling_mean = out["close"].rolling(20).mean()
        rolling_std = out["close"].rolling(20).std()
        out["bb_bw"] = (2 * rolling_std) / (rolling_mean + 1e-12)
        out["bb_pctb"] = (out["close"] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std + 1e-12)

    out["hv20"] = out["log_ret"].rolling(20).std()

    out["range_hl_pct"] = (out["high"] - out["low"]) / (out["close"].abs() + 1e-12)
    out["range_co_pct"] = (out["close"] - out["open"]).abs() / (out["close"].abs() + 1e-12)

    out["volume_z20"] = _pct_zscore(out["volume"].astype(float), 20)

    if PANDAS_TA_AVAILABLE and ta is not None:
        # OBV with error handling
        try:
            obv = ta.obv(out["close"], out["volume"])
            out["obv"] = obv.astype(float) if obv is not None else 0.0
            out["obv_slope10"] = _rolling_slope(out["obv"], 10)
        except Exception as e:
            print(f"[WARN] OBV calculation failed: {e}")
            out["obv"] = 0.0
            out["obv_slope10"] = 0.0

        # MFI with error handling
        try:
            mfi_result = ta.mfi(out["high"], out["low"], out["close"], out["volume"], length=14)
            out["mfi14"] = mfi_result.astype(float) if mfi_result is not None else 0.0
        except Exception as e:
            print(f"[WARN] MFI calculation failed: {e}")
            out["mfi14"] = 0.0
    else:
        # Manual indicators when pandas-ta not available
        out["obv"] = 0.0
        out["obv_slope10"] = 0.0
        out["mfi14"] = 0.0

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

        # ADX with error handling
        try:
            from pandas_ta import adx
            adx_df = adx(out["high"], out["low"], out["close"], length=14)
            if adx_df is not None and not adx_df.empty:
                # Find ADX columns dynamically
                adx_cols = [col for col in adx_df.columns if 'ADX' in col]
                if adx_cols:
                    adx_col = next(col for col in adx_cols if 'ADX' in col and not any(x in col for x in ['p', 'n']))
                    out["adx14"] = adx_df[adx_col].astype(float)
                    out["trend_regime"] = (out["adx14"] > 25).astype(int)
                else:
                    out["adx14"] = 0.0
                    out["trend_regime"] = 0
            else:
                out["adx14"] = 0.0
                out["trend_regime"] = 0
        except Exception as e:
            print(f"[WARN] ADX calculation failed: {e}")
            out["adx14"] = 0.0
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