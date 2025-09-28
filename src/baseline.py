#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Models for Stock Price Prediction
- Naive (Random Walk)
- SMA/EMA with different windows
- ARIMA/Holt-Winters (Statistical models)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

from utils import walk_forward_splits, save_json
from features import _load_prices

class BaselineModel:
    """Base class for baseline models"""

    def __init__(self, name: str):
        self.name = name

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict next day prices based on historical prices"""
        raise NotImplementedError

    def fit(self, prices: np.ndarray):
        """Fit model on training data (for statistical models)"""
        pass

class NaiveModel(BaselineModel):
    """Naive/Random Walk model"""

    def __init__(self):
        super().__init__("Naive")

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict next price = current price"""
        return prices[-1:]

class SMAModel(BaselineModel):
    """Simple Moving Average model"""

    def __init__(self, window: int):
        super().__init__(f"SMA_{window}")
        self.window = window

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict using SMA"""
        return sma_predictor(prices, self.window)

class EMAModel(BaselineModel):
    """Exponential Moving Average model"""

    def __init__(self, window: int):
        super().__init__(f"EMA_{window}")
        self.window = window

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict using EMA"""
        return ema_predictor(prices, self.window)

def naive_predictor(prices: np.ndarray) -> np.ndarray:
    """Naive/Random Walk: predict next price = current price"""
    return prices[-1:]  # Return last price as prediction

def sma_predictor(prices: np.ndarray, window: int) -> np.ndarray:
    """Simple Moving Average predictor"""
    if len(prices) < window:
        return prices[-1:]  # Not enough data, use last price
    sma = np.mean(prices[-window:])
    return np.array([sma])

def ema_predictor(prices: np.ndarray, window: int, alpha: Optional[float] = None) -> np.ndarray:
    """Exponential Moving Average predictor"""
    if len(prices) < window:
        return prices[-1:]  # Not enough data, use last price

    if alpha is None:
        alpha = 2 / (window + 1)  # Standard EMA alpha

    ema_values = []
    for i in range(len(prices)):
        if i == 0:
            ema_values.append(prices[0])
        else:
            ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)

    return np.array([ema_values[-1]])

class ARIMAModel(BaselineModel):
    """ARIMA baseline model"""

    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        super().__init__("ARIMA")
        self.order = order
        self.model = None

    def fit(self, prices: np.ndarray):
        """Fit ARIMA model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMA")

        # Convert to pandas Series for ARIMA
        series = pd.Series(prices)
        try:
            self.model = ARIMA(series, order=self.order)
            self.fitted_model = self.model.fit()
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            # Fallback to naive
            pass

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict next price using fitted ARIMA"""
        if self.model is None or not hasattr(self, 'fitted_model'):
            return naive_predictor(prices)

        try:
            # Forecast next step
            forecast = self.fitted_model.forecast(steps=1)
            return forecast.values
        except Exception as e:
            print(f"ARIMA prediction failed: {e}")
            return naive_predictor(prices)

class HoltWintersModel(BaselineModel):
    """Holt-Winters Exponential Smoothing"""

    def __init__(self, trend: str = 'add', seasonal: str = 'add', seasonal_periods: int = 252):
        super().__init__("HoltWinters")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, prices: np.ndarray):
        """Fit Holt-Winters model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Holt-Winters")

        # Convert to pandas Series
        series = pd.Series(prices)
        try:
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            self.fitted_model = self.model.fit()
        except Exception as e:
            print(f"Holt-Winters fitting failed: {e}")
            # Fallback to naive
            pass

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict next price using fitted Holt-Winters"""
        if self.model is None or not hasattr(self, 'fitted_model'):
            return naive_predictor(prices)

        try:
            # Forecast next step
            forecast = self.fitted_model.forecast(steps=1)
            return forecast.values
        except Exception as e:
            print(f"Holt-Winters prediction failed: {e}")
            return naive_predictor(prices)

def create_baseline_models() -> List[BaselineModel]:
    """Create all baseline models"""
    models = []

    # Naive model
    models.append(NaiveModel())

    # SMA models with different windows
    for window in [5, 10, 20]:
        models.append(SMAModel(window))

    # EMA models with different windows
    for window in [5, 10, 20]:
        models.append(EMAModel(window))

    # Statistical models
    if STATSMODELS_AVAILABLE:
        models.append(ARIMAModel(order=(5, 1, 0)))
        models.append(HoltWintersModel())

    return models

def evaluate_baseline_one_fold(model: BaselineModel, prices: np.ndarray, cfg: dict, fold: int) -> Dict:
    """
    Evaluate baseline model on one fold
    Returns metrics in the same format as DL models
    """
    n = len(prices)
    splits = list(walk_forward_splits(n, cfg["val_size"], cfg["test_size"], cfg["min_train_size"]))

    if fold - 1 >= len(splits):
        raise ValueError(f"Fold {fold} not available")

    tr_slice, va_slice, te_slice = splits[fold - 1]

    # Get train/test data
    train_prices = prices[tr_slice]
    test_prices = prices[te_slice]

    # For statistical models, fit on training data
    if hasattr(model, 'fit'):
        try:
            model.fit(train_prices)
        except Exception as e:
            print(f"Model fitting failed for {model.name}: {e}")

    # Make predictions
    if len(test_prices) == 0:
        # Handle empty test set
        return {
            "fold": fold,
            "model": model.name,
            "rmse": float('nan'),
            "mae": float('nan'),
            "mape": float('nan'),
            "smape": float('nan')
        }

    predictions = []
    actuals = []

    for i in range(len(test_prices)):
        # Use all available history up to current point
        current_prices = np.concatenate([train_prices, test_prices[:i]])
        if len(current_prices) == 0:
            current_prices = test_prices[:i+1]

        try:
            pred = model.predict(current_prices)
            predictions.append(pred[0])
            actuals.append(test_prices[i])
        except Exception as e:
            print(f"Prediction failed for {model.name} at step {i}: {e}")
            # Use naive prediction as fallback
            predictions.append(current_prices[-1] if len(current_prices) > 0 else test_prices[i])
            actuals.append(test_prices[i])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rmse_val = np.sqrt(mean_squared_error(actuals, predictions))
    mae_val = mean_absolute_error(actuals, predictions)

    # MAPE calculation (avoid division by zero)
    mask = np.abs(actuals) > 1e-12
    if mask.sum() == 0:
        mape_val = float('nan')
    else:
        mape_val = np.mean(np.abs((predictions[mask] - actuals[mask]) / actuals[mask])) * 100.0

    # SMAPE calculation
    denom = (np.abs(actuals) + np.abs(predictions)) / 2.0 + 1e-12
    smape_val = np.mean(np.abs(predictions - actuals) / denom) * 100.0

    return {
        "fold": fold,
        "model": model.name,
        "rmse": float(rmse_val),
        "mae": float(mae_val),
        "mape": float(mape_val),
        "smape": float(smape_val)
    }

def run_baseline_evaluation(ticker: str, cfg: dict) -> None:
    """
    Run baseline evaluation for a ticker using walk-forward validation
    """
    print(f"Running baseline evaluation for {ticker}")

    # Load price data
    data_dir = Path("data")
    prices_path = data_dir / ticker / "prices_daily.csv"

    if not prices_path.exists():
        print(f"Price data not found for {ticker} at {prices_path}")
        return

    df = _load_prices(prices_path)
    prices = df["close"].values.astype(float)

    # Create baseline models
    baseline_models = create_baseline_models()

    # Create output directory
    out_dir = Path("reports") / ticker / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation for each model
    all_metrics = []

    for model in baseline_models:
        print(f"Evaluating {model.name}...")

        model_metrics = []
        n_splits = cfg.get("n_splits", 3)

        for fold in range(1, n_splits + 1):
            try:
                metrics = evaluate_baseline_one_fold(model, prices, cfg, fold)
                model_metrics.append(metrics)
                print(f"  Fold {fold}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
            except Exception as e:
                print(f"  Fold {fold} failed: {e}")
                # Add NaN metrics for failed fold
                model_metrics.append({
                    "fold": fold,
                    "model": model.name,
                    "rmse": float('nan'),
                    "mae": float('nan'),
                    "mape": float('nan'),
                    "smape": float('nan')
                })

        all_metrics.extend(model_metrics)

        # Save individual model results
        model_out_dir = out_dir / model.name.lower().replace('_', '')
        model_out_dir.mkdir(exist_ok=True)
        save_json(model_out_dir / "metrics.json", model_metrics)

    # Save all baseline metrics
    save_json(out_dir / "metrics.json", all_metrics)

    # Create summary statistics
    metrics_df = pd.DataFrame(all_metrics)
    summary = metrics_df.groupby("model").agg({
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"],
        "mape": ["mean", "std"],
        "smape": ["mean", "std"]
    }).round(4)

    print("\nBaseline Model Performance Summary:")
    print(summary)

    # Save summary
    summary.to_csv(out_dir / "summary.csv")

    print(f"Baseline evaluation completed for {ticker}")

def main():
    """Main function to run baseline evaluation for all tickers"""
    import yaml

    # Load config
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("❌ Config file not found")
        return

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    # Run for each ticker
    for ticker in cfg["tickers"]:
        run_baseline_evaluation(ticker, cfg)

if __name__ == "__main__":
    main()
