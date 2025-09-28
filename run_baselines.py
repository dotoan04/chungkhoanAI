#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run baseline model evaluation
Usage: python run_baselines.py --tickers FPT HPG VNM
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from baseline import run_baseline_evaluation

def main():
    """Main function to run baseline evaluation"""
    parser = argparse.ArgumentParser(description="Run baseline model evaluation")
    parser.add_argument("--tickers", nargs="+", default=["FPT", "HPG", "VNM"],
                        help="List of tickers to evaluate (default: FPT HPG VNM)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file (default: configs/config.yaml)")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    # Override tickers in config if specified
    if args.tickers:
        cfg["tickers"] = args.tickers

    print("🚀 Starting baseline model evaluation...")
    print(f"📊 Tickers: {cfg['tickers']}")

    # Run for each ticker
    for ticker in cfg["tickers"]:
        run_baseline_evaluation(ticker, cfg)

    print("✅ Baseline evaluation completed!")

if __name__ == "__main__":
    main()
