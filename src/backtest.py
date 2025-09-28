#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, yaml, json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from utils import save_json

def max_drawdown(equity: np.ndarray) -> float:
    high = np.maximum.accumulate(equity)
    dd = (equity - high) / (high + 1e-12)
    return float(dd.min())

def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, ann_factor: int = 252) -> float:
    ex = returns - rf/ann_factor
    return float(np.mean(ex) / (np.std(ex) + 1e-12) * np.sqrt(ann_factor))

def backtest_series(y_true: np.ndarray, y_pred: np.ndarray, thresh=0.0, fee=0.001):
    pos = (y_pred > thresh).astype(int)
    trade = np.abs(np.diff(np.r_[0, pos]))
    pnl = pos * y_true - fee * trade
    equity = np.cumprod(1.0 + pnl)
    return {
        "pnl": pnl,
        "equity": equity,
        "trades": trade,
        "winrate": float((pnl > 0).mean()),
        "turnover": float(trade.mean())
    }

def run_backtest(ticker: str, model: str, lookback: int, thresh: float, fee: float, use_ensemble: bool = False):
    if use_ensemble:
        rep_dir = Path("reports") / ticker / "ensemble"
        preds_file = "predictions.json"
    else:
        rep_dir = Path("reports") / ticker / model
        preds_file = "preds.json"
    
    if not (rep_dir / preds_file).exists():
        print(f"[WARN] Predictions not found: {rep_dir / preds_file}")
        return
        
    preds = json.loads(open(rep_dir / preds_file,"r",encoding="utf-8").read())

    y_true_all, y_pred_all = [], []
    y_true_cls_all, y_pred_cls_all = [], []
    
    # Check if multi-task
    is_multitask = 'y_true_reg' in preds[0] if preds else False
    
    for item in sorted(preds, key=lambda x: x["fold"]):
        if is_multitask:
            y_true_all.extend(item["y_true_reg"])
            y_pred_all.extend(item["y_pred_reg"])
            y_true_cls_all.extend(item["y_true_cls"])
            y_pred_cls_all.extend(item["y_pred_cls"])
        else:
            y_true_all.extend(item["y_true"])
            y_pred_all.extend(item["y_pred"])
    
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    
    if is_multitask:
        y_true_cls = np.array(y_true_cls_all)
        y_pred_cls = np.array(y_pred_cls_all)
        
        # Use classification predictions for trading signals if available
        bt = backtest_series(y_true, y_pred, thresh=thresh, fee=fee)
        bt_cls = backtest_series(y_true, y_pred_cls - 0.5, thresh=0.0, fee=fee)  # Convert probs to signals
        
        out = {
            "regression": {
                "sharpe": sharpe_ratio(bt["pnl"]),
                "max_drawdown": max_drawdown(bt["equity"]),
                "cumulative_return": float(bt["equity"][-1] - 1.0),
                "winrate": bt["winrate"],
                "turnover": bt["turnover"]
            },
            "classification": {
                "sharpe": sharpe_ratio(bt_cls["pnl"]),
                "max_drawdown": max_drawdown(bt_cls["equity"]),
                "cumulative_return": float(bt_cls["equity"][-1] - 1.0),
                "winrate": bt_cls["winrate"],
                "turnover": bt_cls["turnover"],
                "accuracy": float(np.mean((y_pred_cls > 0.5) == y_true_cls))
            }
        }
        
        # Save both regression and classification results
        save_json(rep_dir / "backtest_summary.json", out)
        
        # Save detailed curves
        pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred, 
            "y_true_cls": y_true_cls, "y_pred_cls": y_pred_cls,
            "pnl_reg": bt["pnl"], "equity_reg": bt["equity"],
            "pnl_cls": bt_cls["pnl"], "equity_cls": bt_cls["equity"]
        }).to_csv(rep_dir / "backtest_curve.csv", index=False)
        
        # Plot both equity curves
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(bt["equity"], label='Regression', alpha=0.8)
        plt.plot(bt_cls["equity"], label='Classification', alpha=0.8)
        plt.title(f"{ticker} {model} equity comparison (thresh={thresh}, fee={fee})")
        plt.legend()
        plt.ylabel("Equity (base=1.0)")
        
        plt.subplot(2, 1, 2)
        plt.plot(np.cumsum(bt["pnl"]), label='Regression PnL', alpha=0.8)
        plt.plot(np.cumsum(bt_cls["pnl"]), label='Classification PnL', alpha=0.8)
        plt.xlabel("t (test steps across folds)")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.tight_layout()
        plt.savefig(rep_dir / "backtest_equity.png", dpi=160)
        plt.close()
        
    else:
        # Single-task backtest (original logic)
        bt = backtest_series(y_true, y_pred, thresh=thresh, fee=fee)

        out = {
            "sharpe": sharpe_ratio(bt["pnl"]),
            "max_drawdown": max_drawdown(bt["equity"]),
            "cumulative_return": float(bt["equity"][-1] - 1.0),
            "winrate": bt["winrate"],
            "turnover": bt["turnover"]
        }
        save_json(rep_dir / "backtest_summary.json", out)
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "pnl": bt["pnl"], "equity": bt["equity"]}).to_csv(rep_dir / "backtest_curve.csv", index=False)

        plt.figure()
        plt.plot(bt["equity"])
        plt.title(f"{ticker} {model} equity (thresh={thresh}, fee={fee})")
        plt.xlabel("t (test steps across folds)")
        plt.ylabel("Equity (base=1.0)")
        plt.tight_layout()
        plt.savefig(rep_dir / "backtest_equity.png", dpi=160)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ensemble", action="store_true", help="Backtest ensemble predictions")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    lookback = int(cfg.get("lookback", 60))

    if not cfg.get("backtest", {}).get("enable", True):
        print("[INFO] Backtest disabled in config")
        return

    for t in cfg["tickers"]:
        if args.ensemble:
            # Backtest ensemble predictions
            run_backtest(t, "ensemble", lookback, cfg["backtest"]["thresh"], cfg["backtest"]["fee"], use_ensemble=True)
        else:
            # Backtest individual models
            for m in cfg["models"]:
                run_backtest(t, m, lookback, cfg["backtest"]["thresh"], cfg["backtest"]["fee"])

if __name__ == "__main__":
    main()