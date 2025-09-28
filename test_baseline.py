#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for baseline models
"""

import sys
sys.path.append('src')

from baseline import create_baseline_models, run_baseline_evaluation
import numpy as np

def test_baseline_models():
    """Test baseline models with sample data"""
    print("Testing baseline models...")

    # Create sample price data
    np.random.seed(42)
    sample_prices = np.cumsum(np.random.randn(100)) + 100  # Random walk around 100

    # Create models
    models = create_baseline_models()

    print(f"Created {len(models)} baseline models:")
    for model in models:
        print(f"  - {model.name}")

    # Test each model
    for model in models:
        try:
            if hasattr(model, 'fit'):
                model.fit(sample_prices[:-10])  # Fit on first 90 points

            pred = model.predict(sample_prices[:-10])  # Predict using first 90 points
            print(f"  {model.name}: prediction = {pred[0]:.2f}")
        except Exception as e:
            print(f"  {model.name}: failed - {e}")

def test_full_evaluation():
    """Test full evaluation with config"""
    print("\nTesting full baseline evaluation...")

    # Create minimal config
    cfg = {
        "n_splits": 2,
        "val_size": 10,
        "test_size": 10,
        "min_train_size": 30
    }

    # Create sample price data
    np.random.seed(42)
    sample_prices = np.cumsum(np.random.randn(100)) + 100

    # Test with one model
    from baseline import NaiveModel
    model = NaiveModel()

    try:
        from baseline import evaluate_baseline_one_fold
        metrics = evaluate_baseline_one_fold(model, sample_prices, cfg, fold=1)
        print("Full evaluation test passed!")
        print(f"  Metrics: {metrics}")
    except Exception as e:
        print(f"Full evaluation test failed: {e}")

if __name__ == "__main__":
    test_baseline_models()
    test_full_evaluation()
    print("\nAll tests completed!")
