#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script to run baseline comparison
Usage: python demo_baseline_comparison.py --ticker FPT
"""

import argparse
import sys
sys.path.append('src')

from compare_baselines import print_comparison_table

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare baseline vs DL models")
    parser.add_argument("--ticker", type=str, default="FPT",
                        help="Ticker to compare (default: FPT)")

    args = parser.parse_args()

    print("Comparing baseline models with deep learning models...")
    print_comparison_table([args.ticker])

if __name__ == "__main__":
    main()
