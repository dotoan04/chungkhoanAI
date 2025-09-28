#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script to run baseline models on real data
Usage: python demo_baseline.py --ticker FPT
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from baseline import run_baseline_evaluation

def main():
    """Main function to run baseline evaluation for one ticker"""
    parser = argparse.ArgumentParser(description="Run baseline model evaluation for one ticker")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Ticker to evaluate (e.g., FPT)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file (default: configs/config.yaml)")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    # Override tickers in config to only include specified ticker
    cfg["tickers"] = [args.ticker]

    print(f"Running baseline evaluation for {args.ticker}...")

    # Run for the specified ticker
    run_baseline_evaluation(args.ticker, cfg)

    print(f"Baseline evaluation completed for {args.ticker}!")

if __name__ == "__main__":
    main()
