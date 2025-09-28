#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from features import _load_prices, compute_indicators, add_market_features, finalize_features, build_supervised, windowize
from utils import walk_forward_splits

def prepare_for_ticker(ticker: str, data_dir: Path, lookback: int, cfg, use_multitask: bool = False, predict_price: bool = False) -> None:
    tdir = data_dir / ticker
    prices_path = tdir / "prices_daily.csv"
    df = _load_prices(prices_path)
    df = compute_indicators(df)

    idx_df = None
    if cfg["features"].get("use_market_context", True):
        idx_sym = cfg.get("index_symbol", None)
        if idx_sym:
            idx_path = data_dir / idx_sym / "prices_daily.csv"
            if idx_path.exists():
                idx_df = _load_prices(idx_path)
            else:
                print(f"[WARN] Index data not found at {idx_path}. Skip market features.")
        df = add_market_features(df, idx_df)

    df = finalize_features(df, use_calendar=cfg["features"].get("use_calendar", True),
                           use_regime=cfg["features"].get("use_regime_flags", True))

    # Check if we should predict price instead of returns (for baseline comparison)
    predict_price = cfg.get("predict_price", False)
    df = build_supervised(df, predict_price=predict_price)
    
    # Add directional target for multi-task learning
    if use_multitask:
        df["y_direction"] = (df["y"] > 0).astype(float)  # 1 if up, 0 if down

    date_col = df["date"].values
    num_df = df.select_dtypes(include=[np.number]).copy()
    drop_cols = ["hhv252","llv252","rolling_max_60","close_lag1","open_lag0"]
    for c in drop_cols:
        if c in num_df.columns:
            num_df.drop(columns=[c], inplace=True)

    features = num_df.drop(columns=["y", "y_direction"], errors="ignore").values.astype(float)
    target = num_df["y"].values.astype(float) if "y" in num_df.columns else df["y"].values.astype(float)
    
    if use_multitask:
        target_direction = num_df["y_direction"].values.astype(float) if "y_direction" in num_df.columns else df["y_direction"].values.astype(float)
        # Combine regression and classification targets
        target_combined = np.column_stack([target, target_direction])
    else:
        target_combined = target

    outdir = Path("datasets") / ticker / f"L{lookback}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    if use_multitask:
        pd.DataFrame({"date": date_col, "y": target, "y_direction": target_direction}).to_csv(
            outdir / "target_dates.csv", index=False)
    else:
        pd.DataFrame({"date": date_col, "y": target}).to_csv(outdir / "target_dates.csv", index=False)

    n = len(target)
    splits = list(walk_forward_splits(n, cfg["val_size"], cfg["test_size"], cfg["min_train_size"]))
    if cfg["n_splits"]:
        splits = splits[:cfg["n_splits"]]
    for fold, (tr, va, te) in enumerate(splits, start=1):
        X_tr, y_tr = windowize(features[tr], target_combined[tr], lookback)
        X_va, y_va = windowize(features[va], target_combined[va], lookback)
        X_te, y_te = windowize(features[te], target_combined[te], lookback)
        np.savez_compressed(outdir / f"fold{fold}.npz",
                            X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te)
        print(f"[OK] {ticker} fold{fold}: X_tr {X_tr.shape}, y_tr {y_tr.shape}, X_va {X_va.shape}, X_te {X_te.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--multitask", action="store_true", help="Generate targets for multi-task learning")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    data_dir = Path("data")
    lookback = int(cfg.get("lookback", 60))
    
    # Check if TCN model is being used (implies multi-task)
    use_multitask = args.multitask or "tcn" in cfg.get("models", [])

    # Get predict_price setting from config
    predict_price = cfg.get("predict_price", False)

    # Support for ensemble with multiple lookback windows
    lookbacks = cfg.get("ensemble_lookbacks", [lookback])
    if lookback not in lookbacks:
        lookbacks.append(lookback)

    for t in cfg["tickers"]:
        for lb in lookbacks:
            print(f"Preparing {t} with lookback {lb}...")
            prepare_for_ticker(t, data_dir, lb, cfg, use_multitask, predict_price)

if __name__ == "__main__":
    main()