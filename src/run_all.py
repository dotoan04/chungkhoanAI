#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Run the whole pipeline sequentially with one command.
import subprocess, sys, yaml
from pathlib import Path

def run(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)

def main():
    cfg_path = Path("configs/config.yaml")
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    tickers = cfg["tickers"] + ([cfg.get("index_symbol")] if cfg.get("index_symbol") else [])
    tickers = [t for t in tickers if t]

    run([sys.executable, "src/collect_vnstock.py", "--tickers", *tickers,
         "--start", cfg["start_date"], "--end", cfg["end_date"], "--out", "data", "--source", cfg.get("source","VCI")])

    run([sys.executable, "src/prepare_dataset.py", "--config", str(cfg_path), "--multitask"])

    run([sys.executable, "src/train.py", "--config", str(cfg_path), "--ensemble", "--gpu"])

    run([sys.executable, "src/ensemble.py", "--config", str(cfg_path)])

    run([sys.executable, "src/backtest.py", "--config", str(cfg_path)])
    
    run([sys.executable, "src/backtest.py", "--config", str(cfg_path), "--ensemble"])

if __name__ == "__main__":
    main()