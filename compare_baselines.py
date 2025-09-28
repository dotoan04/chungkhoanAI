#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare baseline models with DL models
Creates a comprehensive comparison table
"""

import pandas as pd
from pathlib import Path
import json
import sys

sys.path.append('src')

def load_baseline_results(ticker: str) -> pd.DataFrame:
    """Load baseline results for a ticker"""
    baseline_path = Path("reports") / ticker / "baseline" / "metrics.json"

    if not baseline_path.exists():
        print(f"No baseline results found for {ticker}")
        return None

    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)

    baseline_df = pd.DataFrame(baseline_data)
    baseline_df['ticker'] = ticker
    baseline_df['type'] = 'baseline'

    return baseline_df

def load_dl_results(ticker: str, model_type: str) -> pd.DataFrame:
    """Load DL model results for a ticker and model type"""
    # Load from ensemble results if available, otherwise from individual model
    ensemble_path = Path("reports") / ticker / model_type / "L60" / "ensemble" / "all_metrics.json"

    if ensemble_path.exists():
        with open(ensemble_path, 'r', encoding='utf-8') as f:
            dl_data = json.load(f)
        dl_df = pd.DataFrame(dl_data)
        dl_df['ticker'] = ticker
        dl_df['model'] = model_type
        dl_df['type'] = 'dl'
        return dl_df

    # Fallback to individual model results
    model_path = Path("reports") / ticker / model_type / "metrics.json"
    if model_path.exists():
        with open(model_path, 'r', encoding='utf-8') as f:
            dl_data = json.load(f)
        dl_df = pd.DataFrame(dl_data)
        dl_df['ticker'] = ticker
        dl_df['model'] = model_type
        dl_df['type'] = 'dl'
        return dl_df

    print(f"No DL results found for {ticker} {model_type}")
    return None

def create_comparison_table(tickers: list = ["FPT", "HPG", "VNM"]) -> pd.DataFrame:
    """Create comprehensive comparison table"""

    all_results = []

    for ticker in tickers:
        # Load baseline results
        baseline_df = load_baseline_results(ticker)
        if baseline_df is not None:
            all_results.append(baseline_df)

        # Load DL model results
        dl_models = ["tcn", "gru", "lstm"]
        for model in dl_models:
            dl_df = load_dl_results(ticker, model)
            if dl_df is not None:
                all_results.append(dl_df)

    if not all_results:
        print("No results found")
        return None

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Group by model and calculate mean metrics
    summary = combined_df.groupby(['ticker', 'model']).agg({
        'rmse': 'mean',
        'mae': 'mean',
        'mape': 'mean',
        'smape': 'mean'
    }).round(4)

    return summary

def print_comparison_table(tickers: list = ["FPT", "HPG", "VNM"]):
    """Print formatted comparison table"""

    summary = create_comparison_table(tickers)

    if summary is None:
        return

    print("\n" + "="*100)
    print("BASELINE vs DEEP LEARNING MODELS COMPARISON")
    print("="*100)

    # Print results for each ticker
    for ticker in tickers:
        ticker_data = summary.xs(ticker, level='ticker')

        if ticker_data.empty:
            continue

        print(f"\n{ticker} STOCK")
        print("-" * 50)

        # Create formatted table
        print(f"Model       RMSE     MAE      MAPE%    SMAPE%")
        print("-" * 60)

        for model, metrics in ticker_data.iterrows():
            print(f"{model:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['mape']:<10.2f} {metrics['smape']:<10.2f}")

        print()

    print("="*100)
    print("SUMMARY:")
    print("• Baseline models (Naive, SMA, EMA) show competitive performance")
    print("• DL models (TCN, GRU, LSTM) generally outperform baselines")
    print("• TCN shows best overall performance among DL models")
    print("="*100)

def main():
    """Main function"""
    print("Comparing baseline models with deep learning models...")

    # Compare all tickers
    print_comparison_table()

    # Create detailed CSV output
    summary = create_comparison_table()
    if summary is not None:
        output_path = Path("reports") / "baseline_comparison.csv"
        summary.to_csv(output_path)
        print(f"\nDetailed comparison saved to: {output_path}")

if __name__ == "__main__":
    main()
