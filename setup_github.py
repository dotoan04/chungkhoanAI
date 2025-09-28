#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup GitHub repository
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            return True
        else:
            print(f"FAILED: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def setup_github():
    """Setup GitHub repository"""

    print("Setting up GitHub repository...")

    # Check if git is initialized
    if not Path(".git").exists():
        print("Initializing git repository...")
        if not run_command("git init", "Initialize git repository"):
            return False

    # Add remote origin (you need to replace with your GitHub repo URL)
    print("Add your GitHub repository URL:")
    print("Example: git remote add origin https://github.com/your-username/chungkhoan-ai.git")
    print("Then run: git push -u origin main")

    # Show current status
    run_command("git status", "Show git status")

    # Add all files
    run_command("git add .", "Add all files to staging")

    # Commit
    run_command('git commit -m "Initial commit: Vietnamese Stock Price Prediction with TCN-Residual"', "Initial commit")

    print("\nGitHub setup completed!")
    print("Next steps:")
    print("1. Create a new repository on GitHub")
    print("2. Copy the repository URL")
    print("3. Run: git remote add origin <YOUR_REPO_URL>")
    print("4. Run: git push -u origin main")

def main():
    """Main function"""
    setup_github()

if __name__ == "__main__":
    main()
