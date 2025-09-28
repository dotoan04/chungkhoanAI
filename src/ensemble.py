#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble predictions from multiple models, seeds, and window sizes.
Implements the ensemble strategy from the TCN-Residual specification:
- 5 seeds × 2 window lengths (L=60,90) × 2 architectures (TCN+GRU)
- Ensemble average for regression; majority vote for classification
"""
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from utils import save_json
from evaluate import rmse, mae, mape, smape

def load_predictions(reports_dir: Path, ticker: str, model: str, lookbacks: List[int], seeds: List[int]) -> Dict[str, Any]:
    """Load predictions from all ensemble components"""
    all_preds = []
    
    for lb in lookbacks:
        for seed in seeds:
            pred_path = reports_dir / ticker / model / f"L{lb}" / f"seed{seed}" / "preds.json"
            if pred_path.exists():
                with open(pred_path, 'r') as f:
                    preds = json.load(f)
                    for pred in preds:
                        pred['lookback'] = lb
                        pred['seed'] = seed
                        pred['model'] = model
                        all_preds.append(pred)
            else:
                print(f"[WARN] Missing predictions: {pred_path}")
    
    return all_preds

def ensemble_predictions(all_preds: List[Dict], method: str = "average") -> Dict[str, Any]:
    """Ensemble predictions using specified method"""
    # Group by fold
    fold_groups = {}
    for pred in all_preds:
        fold = pred['fold']
        if fold not in fold_groups:
            fold_groups[fold] = []
        fold_groups[fold].append(pred)
    
    ensemble_preds = []
    ensemble_metrics = []
    
    for fold, preds in fold_groups.items():
        # Filter out predictions with None values or missing keys
        valid_preds = []
        for pred in preds:
            if 'y_true_reg' in pred and 'y_pred_reg' in pred:
                if (pred['y_true_reg'] is not None and pred['y_pred_reg'] is not None and
                    len(pred['y_true_reg']) > 0 and len(pred['y_pred_reg']) > 0):
                    valid_preds.append(pred)
            elif 'y_true' in pred and 'y_pred' in pred:
                if (pred['y_true'] is not None and pred['y_pred'] is not None and
                    len(pred['y_true']) > 0 and len(pred['y_pred']) > 0):
                    valid_preds.append(pred)
        
        if not valid_preds:
            print(f"[WARN] No valid predictions found for fold {fold}")
            continue
            
        # Group by prediction length to handle different lookback windows
        length_groups = {}
        for pred in valid_preds:
            if 'y_pred_reg' in pred:
                pred_len = len(pred['y_pred_reg'])
            else:
                pred_len = len(pred['y_pred'])
            
            if pred_len not in length_groups:
                length_groups[pred_len] = []
            length_groups[pred_len].append(pred)
        
        # Process each length group separately
        for pred_len, group_preds in length_groups.items():
            # Check if multi-task
            is_multitask = 'y_true_reg' in group_preds[0]
            
            if is_multitask:
                # Multi-task ensemble
                y_true_reg = np.array(group_preds[0]['y_true_reg'])
                y_true_cls = np.array(group_preds[0]['y_true_cls']) if group_preds[0]['y_true_cls'] is not None else None
                
                # Ensemble regression predictions (average)
                reg_preds = np.array([pred['y_pred_reg'] for pred in group_preds])
                y_pred_reg_ensemble = np.mean(reg_preds, axis=0)
                
                # Ensemble classification predictions (average probabilities) if available
                cls_preds_valid = [pred['y_pred_cls'] for pred in group_preds if pred['y_pred_cls'] is not None]
                if cls_preds_valid and y_true_cls is not None:
                    cls_preds = np.array(cls_preds_valid)
                    y_pred_cls_ensemble = np.mean(cls_preds, axis=0)
                    
                    # Calculate ensemble metrics
                    metrics = {
                        "fold": fold,
                        "pred_length": pred_len,
                        "rmse": rmse(y_true_reg, y_pred_reg_ensemble),
                        "mae": mae(y_true_reg, y_pred_reg_ensemble),
                        "mape": mape(y_true_reg, y_pred_reg_ensemble),
                        "smape": smape(y_true_reg, y_pred_reg_ensemble),
                        "cls_accuracy": float(np.mean((y_pred_cls_ensemble > 0.5) == y_true_cls)),
                        "cls_precision": float(np.mean(y_true_cls[y_pred_cls_ensemble > 0.5])) if np.sum(y_pred_cls_ensemble > 0.5) > 0 else 0.0,
                        "n_models": len(group_preds)
                    }
                    
                    ensemble_preds.append({
                        "fold": fold,
                        "pred_length": pred_len,
                        "y_true_reg": y_true_reg.tolist(),
                        "y_pred_reg": y_pred_reg_ensemble.tolist(),
                        "y_true_cls": y_true_cls.tolist(),
                        "y_pred_cls": y_pred_cls_ensemble.tolist(),
                        "n_models": len(group_preds)
                    })
                else:
                    # Regression only
                    metrics = {
                        "fold": fold,
                        "pred_length": pred_len,
                        "rmse": rmse(y_true_reg, y_pred_reg_ensemble),
                        "mae": mae(y_true_reg, y_pred_reg_ensemble),
                        "mape": mape(y_true_reg, y_pred_reg_ensemble),
                        "smape": smape(y_true_reg, y_pred_reg_ensemble),
                        "n_models": len(group_preds)
                    }
                    
                    ensemble_preds.append({
                        "fold": fold,
                        "pred_length": pred_len,
                        "y_true_reg": y_true_reg.tolist(),
                        "y_pred_reg": y_pred_reg_ensemble.tolist(),
                        "n_models": len(group_preds)
                    })
                
            else:
                # Single-task ensemble
                y_true = np.array(group_preds[0]['y_true'])
                
                # Ensemble predictions (average)
                pred_array = np.array([pred['y_pred'] for pred in group_preds])
                y_pred_ensemble = np.mean(pred_array, axis=0)
                
                # Calculate ensemble metrics
                metrics = {
                    "fold": fold,
                    "pred_length": pred_len,
                    "rmse": rmse(y_true, y_pred_ensemble),
                    "mae": mae(y_true, y_pred_ensemble),
                    "mape": mape(y_true, y_pred_ensemble),
                    "smape": smape(y_true, y_pred_ensemble),
                    "n_models": len(group_preds)
                }
                
                ensemble_preds.append({
                    "fold": fold,
                    "pred_length": pred_len,
                    "y_true": y_true.tolist(),
                    "y_pred": y_pred_ensemble.tolist(),
                    "n_models": len(group_preds)
                })
            
            ensemble_metrics.append(metrics)
    
    return {
        "predictions": ensemble_preds,
        "metrics": ensemble_metrics
    }

def create_ensemble(ticker: str, models: List[str], lookbacks: List[int], seeds: List[int], reports_dir: Path):
    """Create ensemble from multiple models"""
    print(f"Creating ensemble for {ticker}...")
    
    all_model_preds = []
    for model in models:
        preds = load_predictions(reports_dir, ticker, model, lookbacks, seeds)
        all_model_preds.extend(preds)
    
    if not all_model_preds:
        print(f"[WARN] No predictions found for {ticker}")
        return
    
    # Create ensemble
    ensemble_result = ensemble_predictions(all_model_preds, method="average")
    
    # Save ensemble results
    ensemble_dir = reports_dir / ticker / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(ensemble_dir / "predictions.json", ensemble_result["predictions"])
    save_json(ensemble_dir / "metrics.json", ensemble_result["metrics"])
    
    # Save summary statistics
    df_metrics = pd.DataFrame(ensemble_result["metrics"])
    df_metrics.to_csv(ensemble_dir / "ensemble_summary.csv", index=False)
    
    # Print summary
    print(f"[OK] {ticker} ensemble results:")
    summary_stats = df_metrics.describe()
    print(summary_stats)
    
    return ensemble_result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--models", nargs="+", help="Models to ensemble (default: from config)")
    args = ap.parse_args()
    
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    
    # Ensemble configuration
    ensemble_config = cfg.get("ensemble", {})
    models = args.models or cfg.get("models", ["tcn", "gru"])
    lookbacks = cfg.get("ensemble_lookbacks", [60, 90])
    seeds = ensemble_config.get("seeds", [42, 123, 456, 789, 999])
    
    reports_dir = Path("reports")
    
    for ticker in cfg["tickers"]:
        create_ensemble(ticker, models, lookbacks, seeds, reports_dir)

if __name__ == "__main__":
    main()