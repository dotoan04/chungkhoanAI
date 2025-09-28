#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup unnecessary files for GitHub repository
"""

import os
from pathlib import Path

# List of files to delete
files_to_delete = [
    "BAI_TIEU_LUAN_LY_THUYET.md",
    "check_status.py",
    "ChungKhoanAI.zip",
    "colab_notebook.py",
    "COLAB_README.md",
    "daily_monitor.py",
    "demo_baseline_comparison.py",
    "demo_baseline.py",
    "generate_signals.py",
    "generate_thesis_content.py",
    "get_predictions.py",
    "investment_analysis.py",
    "monitor_training.py",
    "next_steps_guide.py",
    "NEXT_STEPS_SUMMARY.md",
    "paper_trading.py",
    "resume_training.py",
    "test_baseline.py",
    "THESIS_README.md"
]

def cleanup_files():
    """Delete unnecessary files"""
    deleted_count = 0

    for filename in files_to_delete:
        file_path = Path(filename)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        else:
            print(f"File not found: {filename}")

    print(f"\nTotal files deleted: {deleted_count}")

def remove_thesis_content():
    """Remove thesis_content directory (optional)"""
    thesis_dir = Path("thesis_content")
    if thesis_dir.exists():
        try:
            import shutil
            shutil.rmtree(thesis_dir)
            print(f"Removed directory: {thesis_dir}")
        except Exception as e:
            print(f"Error removing {thesis_dir}: {e}")

if __name__ == "__main__":
    print("Cleaning up files for GitHub repository...")
    cleanup_files()
    # remove_thesis_content()  # Uncomment if you want to remove thesis_content
    print("\nCleanup completed!")
