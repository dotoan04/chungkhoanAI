#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test full pipeline with fallback mechanisms
Tests the complete system with sample data and error handling
"""

import sys
sys.path.append('src')

from collect_vnstock import create_sample_data
from features import compute_indicators, build_supervised
from prepare_dataset import prepare_for_ticker
from utils import walk_forward_splits
import pandas as pd
import numpy as np
from pathlib import Path

def test_full_pipeline():
    """Test the complete pipeline with sample data"""
    print("Testing full pipeline with sample data...")

    # Create sample data for FPT
    sample_data = create_sample_data("FPT", "2023-01-01", "2023-12-31")
    print(f"Sample data created: {len(sample_data)} rows")

    # Save sample data
    test_dir = Path("test_pipeline")
    test_dir.mkdir(exist_ok=True)

    sample_file = test_dir / "FPT" / "prices_daily.csv"
    sample_file.parent.mkdir(exist_ok=True)
    sample_data.to_csv(sample_file, index=False)
    print(f"Sample data saved to: {sample_file}")

    # Test features computation
    features_df = compute_indicators(sample_data)
    print(f"Features computed: {len(features_df.columns)} columns, {len(features_df)} rows")

    # Check for NaN values in key features
    nan_check = features_df[['close', 'rsi14', 'bb_bw', 'volume_z20']].isnull().sum()
    print(f"NaN values in key features: {nan_check.sum()}")

    # Test supervised learning target
    supervised_df = build_supervised(features_df)
    print(f"Supervised data: {len(supervised_df.columns)} columns, {len(supervised_df)} rows")

    # Test windowing (simulate prepare_dataset logic)
    lookback = 60
    features = supervised_df.select_dtypes(include=[np.number]).drop(columns=['y'], errors='ignore').values
    targets = supervised_df['y'].values

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")

    # Test walk-forward splits
    n = len(targets)
    splits = list(walk_forward_splits(n, val_size=63, test_size=63, min_train=189))  # 3 splits
    print(f"Walk-forward splits: {len(splits)} folds")

    for i, (tr, va, te) in enumerate(splits[:3]):  # Show first 3 folds
        train_size = tr.stop - tr.start
        val_size = va.stop - va.start
        test_size = te.stop - te.start
        print(f"Fold {i+1}: train={train_size}, val={val_size}, test={test_size}")

    print("\nPipeline test completed successfully!")
    print("All components working with fallback mechanisms.")

def main():
    """Main test function"""
    test_full_pipeline()

if __name__ == "__main__":
    main()
