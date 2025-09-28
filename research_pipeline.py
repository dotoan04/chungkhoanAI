#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎓 CHUNGKHOAN AI - RESEARCH PIPELINE FOR THESIS
Complete pipeline to generate all results for academic research
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
import time
from datetime import datetime

class ResearchPipeline:
    """Complete research pipeline for thesis"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.start_time = time.time()

    def run_command(self, command: str, description: str = ""):
        """Run a command with logging"""
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"{'='*60}")
        print(f"Command: {command}")

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ SUCCESS")
                if result.stdout:
                    print("Output:", result.stdout[-500:])  # Last 500 chars
            else:
                print("❌ FAILED")
                print("Error:", result.stderr)
                return False
            return True
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False

    def check_requirements(self):
        """Check if all requirements are installed"""
        print("\n🔍 Checking requirements...")

        try:
            import pandas as pd
            import numpy as np
            import tensorflow as tf
            import pandas_ta as ta
            import statsmodels
            import vnstock
            import sklearn

            print("✅ All core packages available")

            # Check GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"✅ GPU available: {len(gpus)} device(s)")
            else:
                print("⚠️ No GPU found, using CPU")

            return True

        except ImportError as e:
            print(f"❌ Missing package: {e}")
            print("Run: pip install -r requirements.txt")
            return False

    def step_1_data_collection(self, tickers: list = None):
        """Step 1: Collect data from vnstock"""
        print("\n📊 STEP 1: DATA COLLECTION")

        if tickers is None:
            tickers = ["FPT", "HPG", "VNM", "VNINDEX"]

        command = f"python src/collect_vnstock.py --tickers {' '.join(tickers)} --start 2015-01-01 --end 2025-08-28"
        return self.run_command(command, "Collecting stock data from vnstock API")

    def step_2_prepare_datasets(self):
        """Step 2: Prepare datasets for both baseline and DL models"""
        print("\n🔧 STEP 2: DATASET PREPARATION")

        # Standard DL models (predicting returns)
        cmd1 = "python src/prepare_dataset.py --config configs/config.yaml"
        success1 = self.run_command(cmd1, "Preparing dataset for DL models (returns)")

        # Baseline comparison (predicting prices)
        cmd2 = "python src/prepare_dataset.py --config configs/config_baseline.yaml"
        success2 = self.run_command(cmd2, "Preparing dataset for DL models (prices)")

        return success1 and success2

    def step_3_run_baselines(self):
        """Step 3: Run baseline models evaluation"""
        print("\n📈 STEP 3: BASELINE MODELS EVALUATION")

        command = "python run_baselines.py --tickers FPT HPG VNM"
        return self.run_command(command, "Running baseline models (Naive, SMA, EMA, ARIMA)")

    def step_4_train_dl_models(self):
        """Step 4: Train deep learning models"""
        print("\n🧠 STEP 4: DEEP LEARNING MODEL TRAINING")

        # Train standard models (returns)
        cmd1 = "python src/train.py --config configs/config.yaml"
        success1 = self.run_command(cmd1, "Training DL models (returns prediction)")

        # Train comparison models (prices)
        cmd2 = "python src/train.py --config configs/config_baseline.yaml"
        success2 = self.run_command(cmd2, "Training DL models (price prediction)")

        return success1 and success2

    def step_5_ensemble_models(self):
        """Step 5: Create ensemble predictions"""
        print("\n🤝 STEP 5: ENSEMBLE MODEL CREATION")

        cmd1 = "python src/ensemble.py --config configs/config.yaml"
        success1 = self.run_command(cmd1, "Creating ensemble predictions (returns)")

        cmd2 = "python src/ensemble.py --config configs/config_baseline.yaml"
        success2 = self.run_command(cmd2, "Creating ensemble predictions (prices)")

        return success1 and success2

    def step_6_backtesting(self):
        """Step 6: Run backtesting"""
        print("\n💰 STEP 6: BACKTESTING")

        # Individual models
        cmd1 = "python src/backtest.py --config configs/config.yaml"
        success1 = self.run_command(cmd1, "Backtesting individual models")

        # Ensemble models
        cmd2 = "python src/backtest.py --config configs/config.yaml --ensemble"
        success2 = self.run_command(cmd2, "Backtesting ensemble models")

        return success1 and success2

    def step_7_comparison_analysis(self):
        """Step 7: Create comparison analysis"""
        print("\n📊 STEP 7: COMPARISON ANALYSIS")

        command = "python compare_baselines.py"
        return self.run_command(command, "Creating baseline vs DL model comparison")

    def step_8_generate_reports(self):
        """Step 8: Generate comprehensive research reports"""
        print("\n📋 STEP 8: RESEARCH REPORTS")

        # Generate summary statistics
        cmd1 = "python -c \"import sys; sys.path.append('src'); from research_summary import generate_thesis_summary\""
        success1 = self.run_command(cmd1, "Generating thesis summary")

        # Generate LaTeX tables
        cmd2 = "python -c \"import sys; sys.path.append('src'); from research_summary import generate_latex_tables\""
        success2 = self.run_command(cmd2, "Generating LaTeX tables")

        return success1 and success2

    def run_full_pipeline(self):
        """Run complete research pipeline"""
        print("🎓 CHUNGKHOAN AI - THESIS RESEARCH PIPELINE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check requirements first
        if not self.check_requirements():
            print("❌ Requirements check failed. Please install missing packages.")
            return False

        steps = [
            ("Data Collection", self.step_1_data_collection),
            ("Dataset Preparation", self.step_2_prepare_datasets),
            ("Baseline Models", self.step_3_run_baselines),
            ("DL Model Training", self.step_4_train_dl_models),
            ("Ensemble Creation", self.step_5_ensemble_models),
            ("Backtesting", self.step_6_backtesting),
            ("Comparison Analysis", self.step_7_comparison_analysis),
            ("Research Reports", self.step_8_generate_reports),
        ]

        success_count = 0
        for step_name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
                else:
                    print(f"⚠️ Step '{step_name}' had issues but continuing...")
            except Exception as e:
                print(f"❌ Step '{step_name}' failed with error: {e}")
                print("Continuing with next steps...")

        # Final summary
        end_time = time.time()
        duration = end_time - self.start_time

        print(f"\n{'='*80}")
        print("🎓 PIPELINE COMPLETION SUMMARY")
        print(f"{'='*80}")
        print(f"Total duration: {duration/60".1f"} minutes")
        print(f"Successful steps: {success_count}/{len(steps)}")

        if success_count == len(steps):
            print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
            print("📁 Check the following directories for results:")
            print("  - reports/          # Model performance and metrics")
            print("  - datasets/        # Prepared training data")
            print("  - research_summary.txt  # Thesis-ready summary")
            print("  - latex_tables/    # LaTeX tables for thesis")
        else:
            print(f"⚠️ {len(steps) - success_count} steps had issues")
            print("Check the logs above for details")

        return success_count == len(steps)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run complete research pipeline for thesis")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick version (skip time-consuming steps)")
    parser.add_argument("--tickers", nargs="+", default=["FPT", "HPG", "VNM"],
                        help="Tickers to analyze (default: FPT HPG VNM)")

    args = parser.parse_args()

    pipeline = ResearchPipeline()
    success = pipeline.run_full_pipeline()

    if success:
        print("\n🎓 Ready to write your thesis!")
        print("Key findings and results are available in:")
        print("  - reports/baseline_comparison.csv")
        print("  - reports/research_summary.txt")
        print("  - reports/latex_tables/")
    else:
        print("\n⚠️ Pipeline completed with some issues")
        print("Review the logs and fix any problems before proceeding")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
