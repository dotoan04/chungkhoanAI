#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test features robustness with different pandas-ta versions
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from features import compute_indicators

def create_test_data():
    """Create test OHLCV data"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

    # Create realistic stock data
    close_prices = []
    current_price = 50.0

    for i in range(len(dates)):
        # Random walk with drift
        change = np.random.normal(0.001, 0.02)  # 0.1% daily drift, 2% volatility
        current_price *= (1 + change)
        close_prices.append(max(current_price, 0.1))

    # Create OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        # Generate realistic OHLC from close
        volatility = 0.02
        open_price = close * (1 + np.random.normal(0, volatility/2))
        high_price = max(open_price, close) * (1 + abs(np.random.normal(0, volatility)))
        low_price = min(open_price, close) * (1 - abs(np.random.normal(0, volatility)))
        volume = int(np.random.normal(1000000, 300000))

        data.append({
            'time': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close, 2),
            'volume': volume
        })

    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    return df

def test_features_robustness():
    """Test features calculation with error handling"""
    print("Testing features robustness...")

    # Create test data
    df = create_test_data()
    print(f"Test data shape: {df.shape}")

    try:
        # Test compute_indicators
        result_df = compute_indicators(df)
        print(f"Features computed successfully. Shape: {result_df.shape}")
        print(f"Number of features: {len(result_df.columns)}")

        # Check for NaN values
        nan_counts = result_df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(f"Columns with NaN values: {len(nan_cols)}")
            print(f"Sample NaN counts: {nan_cols.head()}")
        else:
            print("No NaN values found in features!")

        # Show some key features
        key_features = ['close', 'rsi14', 'bb_bw', 'macd', 'volume_z20']
        available_features = [col for col in key_features if col in result_df.columns]
        print(f"\nKey features available: {available_features}")

        if available_features:
            print("Sample values for key features:")
            print(result_df[available_features].tail(3))

        return True

    except Exception as e:
        print(f"Error in features computation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pandas_ta_version():
    """Check pandas-ta version and available functions"""
    print("\nChecking pandas-ta version and functions...")

    try:
        import pandas_ta as ta
        print(f"pandas-ta version: {ta.__version__}")

        # Test some basic functions
        test_df = pd.DataFrame({
            'high': [100, 101, 102, 103, 104],
            'low': [95, 96, 97, 98, 99],
            'close': [98, 99, 100, 101, 102],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        # Test basic indicators
        try:
            sma = ta.sma(test_df['close'], length=3)
            print(f"SMA test: {len(sma)} values")
        except Exception as e:
            print(f"SMA test failed: {e}")

        try:
            bb = ta.bbands(test_df['close'], length=3)
            print(f"Bollinger Bands columns: {list(bb.columns)}")
        except Exception as e:
            print(f"Bollinger Bands test failed: {e}")

        try:
            macd = ta.macd(test_df['close'])
            print(f"MACD columns: {list(macd.columns)}")
        except Exception as e:
            print(f"MACD test failed: {e}")

    except Exception as e:
        print(f"pandas-ta check failed: {e}")

if __name__ == "__main__":
    test_pandas_ta_version()
    success = test_features_robustness()

    if success:
        print("\nFeatures robustness test PASSED!")
        print("The system should work with different pandas-ta versions.")
    else:
        print("\nFeatures robustness test FAILED!")
        print("Need to fix compatibility issues.")
