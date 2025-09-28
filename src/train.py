#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, yaml
import gc
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# GPU Configuration
def setup_gpu():
    """Cấu hình GPU để training tối ưu cho RTX 3050Ti"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth để tránh OOM với 4GB VRAM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"🚀 Using GPU: {gpus[0]}")
            
            # Mixed precision training for RTX cards (tăng tốc + tiết kiệm VRAM)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("🔥 Mixed precision FP16 enabled (RTX optimized)")

            
            return True
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("⚠️ No GPU found, using CPU")
        return False

from utils import seed_everything, fit_transform_scalers, apply_scalers, save_json
from models import make_model, WarmupCosineDecay
from evaluate import rmse, mae, mape, smape, directional_accuracy, information_coefficient, sharpe_like

def _load_dates_for_fold(ds_path: Path, fold: int, lookback: int, cfg):
    import pandas as pd
    dates_df = pd.read_csv(ds_path / "target_dates.csv")
    dates_df["date"] = pd.to_datetime(dates_df["date"])
    n = len(dates_df)
    val_size = int(cfg["val_size"]) ; test_size = int(cfg["test_size"]) ; min_train = int(cfg["min_train_size"])
    # Reconstruct te slice for the given fold index (1-based)
    f = 0
    for t_end in range(min_train, n - (val_size + test_size) + 1, test_size):
        f += 1
        tr = slice(0, t_end)
        va = slice(t_end, t_end + val_size)
        te = slice(t_end + val_size, t_end + val_size + test_size)
        if f == fold:
            start = te.start + lookback - 1
            stop = te.stop
            idx = np.arange(max(start, 0), stop)
            return dates_df.iloc[idx]["date"].reset_index(drop=True)
    return None

def _compute_true_multihorizon_returns(ticker: str, test_dates, horizons):
    import pandas as pd
    price_path = Path("data")/ticker/"prices_daily.csv"
    if not price_path.exists():
        return None
    df = pd.read_csv(price_path)
    df["date"] = pd.to_datetime(df["time"] if "time" in df.columns else df["date"]) ; df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float).values
    date_to_idx = {d:i for i,d in enumerate(df["date"].values)}
    N = len(test_dates)
    y_true = {H: np.full((N,), np.nan, dtype=float) for H in horizons}
    for i, d in enumerate(test_dates.values):
        idx = date_to_idx.get(d, None)
        if idx is None:
            continue
        for H in horizons:
            j = idx + H
            if j < len(close):
                y_true[H][i] = float(np.log(close[j] + 1e-12) - np.log(close[idx] + 1e-12))
    return y_true

def train_one_fold(model_name, ds_path: Path, fold: int, cfg, seed: int = 42, ticker: str = None):
    # Set seed for reproducibility
    seed_everything(seed)
    
    npz = np.load(ds_path / f"fold{fold}.npz")
    X_tr, y_tr = npz["X_tr"], npz["y_tr"]
    X_va, y_va = npz["X_va"], npz["y_va"]
    X_te, y_te = npz["X_te"], npz["y_te"]

    # Clean features to avoid NaN/Inf propagating into training
    def clean_features(X: np.ndarray) -> np.ndarray:
        X2 = X.astype(np.float32, copy=False)
        flat = X2.reshape(-1, X2.shape[-1])
        # Compute per-feature median ignoring NaNs/Inf
        finite_flat = np.where(np.isfinite(flat), flat, np.nan)
        medians = np.nanmedian(finite_flat, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        # Replace non-finite with medians
        non_finite_mask = ~np.isfinite(flat)
        if non_finite_mask.any():
            col_idx = np.where(non_finite_mask)[1]
            flat[non_finite_mask] = medians[col_idx]
        return flat.reshape(X2.shape)

    X_tr = clean_features(X_tr)
    X_va = clean_features(X_va)
    X_te = clean_features(X_te)
    
    # Check if this is multi-task learning (y has 2 dimensions AND model supports multi-task)
    is_multitask = (len(y_tr.shape) > 1 and y_tr.shape[1] == 2 and model_name.lower() == "tcn")
    # Multi-horizon enabled only when configured and TCN multitask
    horizons = cfg.get("horizons", []) or []
    loss_type = cfg.get("loss_type", "huber")
    lambda_cls_cfg = float(cfg.get("lambda_cls", cfg.get("tcn", {}).get("loss_lambda", 0.3)))
    n_mc_dropout = int(cfg.get("n_mc_dropout", 0))
    predict_price = cfg.get("predict_price", False)
    enable_mh = (model_name.lower() == "tcn" and is_multitask and len(horizons) > 0 and not predict_price)
    
    if is_multitask:
        # Split targets: regression and classification  
        y_tr_reg, y_tr_cls = y_tr[:, 0], y_tr[:, 1]
        y_va_reg, y_va_cls = y_va[:, 0], y_va[:, 1]
        y_te_reg, y_te_cls = y_te[:, 0], y_te[:, 1]
        
        # Scale only regression targets, keep classification as is
        scalers, X_trs, y_trs_reg = fit_transform_scalers(X_tr, y_tr_reg)
        X_vas, y_vas_reg = apply_scalers(scalers, X_va, y_va_reg)
        X_tes, y_tes_reg = apply_scalers(scalers, X_te, y_te_reg)
        
        # Combine targets
        if enable_mh:
            H = len(horizons)
            # Backward compatible placeholder labels: tile 1-step labels across horizons
            y_trs_reg_mh = np.tile(y_trs_reg.reshape(-1,1), (1,H))
            y_vas_reg_mh = np.tile(y_vas_reg.reshape(-1,1), (1,H))
            y_tes_reg_mh = np.tile(y_tes_reg.reshape(-1,1), (1,H))
            y_trs_cls_mh = np.tile(y_tr_cls.reshape(-1,1), (1,H))
            y_vas_cls_mh = np.tile(y_va_cls.reshape(-1,1), (1,H))
            y_tes_cls_mh = np.tile(y_te_cls.reshape(-1,1), (1,H))

            y_trs = {"reg": y_trs_reg_mh.astype(np.float32), "cls": y_trs_cls_mh.astype(np.float32)}
            y_vas = {"reg": y_vas_reg_mh.astype(np.float32), "cls": y_vas_cls_mh.astype(np.float32)}
            y_tes = {"reg": y_tes_reg_mh.astype(np.float32), "cls": y_tes_cls_mh.astype(np.float32)}
        else:
            y_trs = np.column_stack([y_trs_reg, y_tr_cls])
            y_vas = np.column_stack([y_vas_reg, y_va_cls])
            y_tes = np.column_stack([y_tes_reg, y_te_cls])
        
        # Ensure correct data types
        y_trs = y_trs.astype(np.float32)
        y_vas = y_vas.astype(np.float32)
        y_tes = y_tes.astype(np.float32)
    else:
        # For single-task models, extract only regression targets if multi-dimensional
        if len(y_tr.shape) > 1 and y_tr.shape[1] == 2:
            y_tr_reg = y_tr[:, 0]
            y_va_reg = y_va[:, 0] 
            y_te_reg = y_te[:, 0]
        else:
            y_tr_reg = y_tr
            y_va_reg = y_va
            y_te_reg = y_te
            
        scalers, X_trs, y_trs = fit_transform_scalers(X_tr, y_tr_reg)
        X_vas, y_vas = apply_scalers(scalers, X_va, y_va_reg)
        X_tes, y_tes = apply_scalers(scalers, X_te, y_te_reg)
        
        # Ensure correct data types
        y_trs = y_trs.astype(np.float32)
        y_vas = y_vas.astype(np.float32)
        y_tes = y_tes.astype(np.float32)
    
    # Ensure X data is float32 and finite after scaling
    def ensure_finite(arr: np.ndarray) -> np.ndarray:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr.astype(np.float32, copy=False)

    X_trs = ensure_finite(X_trs)
    X_vas = ensure_finite(X_vas)
    X_tes = ensure_finite(X_tes)

    # Model configuration
    model_kwargs = {}
    if model_name.lower() == "tcn":
        tcn_config = cfg.get("tcn", {})
        model_kwargs.update({
            "filters": tcn_config.get("filters", 64),
            "kernel_size": tcn_config.get("kernel_size", 5),
            "dilations": tcn_config.get("dilations", [1, 2, 4, 8]),
            "dropout_rate": tcn_config.get("dropout_rate", 0.1),
            "use_se": tcn_config.get("use_se", True),
            "use_multitask": is_multitask,
            "multihorizon": enable_mh,
            "horizons": horizons,
            "loss_type": loss_type,
            "lambda_cls": lambda_cls_cfg
        })
    
    # Learning rate and optimizer settings
    lr = cfg.get("learning_rate", 1e-3)
    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 64)
    patience = cfg.get("patience", 15)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    
    # Create model
    model = make_model(model_name, input_shape=X_trs.shape[1:], lr=lr, **model_kwargs)
    
    # For TCN, optionally use different optimizer settings
    if model_name.lower() == "tcn":
        tcn_config = cfg.get("tcn", {})
        weight_decay = float(tcn_config.get("weight_decay", 1e-4))
        loss_lambda = float(lambda_cls_cfg)
        
        # Simple recompile with AdamW and better settings
        if is_multitask and not enable_mh:
            from models import huber_multi_loss
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=lr, 
                    weight_decay=weight_decay,
                    clipnorm=1.0
                ),
                loss=huber_multi_loss(delta=1.0),
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    clipnorm=1.0
                ),
                loss=tf.keras.losses.Huber(delta=1.0),
                metrics=['mae']
            )

    ckpt_path = ds_path / f"best_{model_name}_fold{fold}_seed{seed}.keras"
    cbs = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience//3, min_lr=1e-6, verbose=0),
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True)
    ]
    
    # Build efficient tf.data pipelines
    AUTOTUNE = tf.data.AUTOTUNE
    if enable_mh and is_multitask:
        ds_tr = tf.data.Dataset.from_tensor_slices((X_trs, y_trs))
    else:
        ds_tr = tf.data.Dataset.from_tensor_slices((X_trs, y_trs))
    ds_tr = ds_tr.shuffle(min(len(X_trs), 4096)).batch(batch_size).prefetch(AUTOTUNE)
    if enable_mh and is_multitask:
        ds_va = tf.data.Dataset.from_tensor_slices((X_vas, y_vas))
    else:
        ds_va = tf.data.Dataset.from_tensor_slices((X_vas, y_vas))
    ds_va = ds_va.batch(batch_size).prefetch(AUTOTUNE)

    print(f"🚀 GPU training with batch_size={batch_size}")

    model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=epochs,
        verbose=0,
        callbacks=cbs
    )

    # Predictions
    ds_te = tf.data.Dataset.from_tensor_slices(X_tes).batch(batch_size).prefetch(AUTOTUNE)
    # MC-Dropout inference if configured and model supports
    if n_mc_dropout and n_mc_dropout > 1 and is_multitask and enable_mh:
        reg_samples = []
        cls_samples = []
        for _ in range(n_mc_dropout):
            out = model.predict(ds_te, verbose=0)
            if isinstance(out, dict) and "reg" in out:
                reg_samples.append(out["reg"])  # [N,H]
                cls_samples.append(out["cls"])  # [N,H]
            else:
                # Fallback: direct call with training=True
                out2 = model(X_tes, training=True)
                reg_samples.append(out2["reg"].numpy())
                cls_samples.append(out2["cls"].numpy())
        reg_samples = np.stack(reg_samples, axis=0)  # [M,N,H]
        cls_samples = np.stack(cls_samples, axis=0)  # [M,N,H]
        y_pred_reg_mean = reg_samples.mean(axis=0)
        y_pred_reg_std = reg_samples.std(axis=0)
        q10 = np.quantile(reg_samples, 0.1, axis=0)
        q90 = np.quantile(reg_samples, 0.9, axis=0)
        y_pred_out = {"reg": y_pred_reg_mean, "cls": cls_samples.mean(axis=0)}
        uncertainty = {"mean": y_pred_reg_mean, "std": y_pred_reg_std, "q10": q10, "q90": q90}
    else:
        y_pred_te = model.predict(ds_te, verbose=0)
        if isinstance(y_pred_te, dict):
            y_pred_out = y_pred_te
            uncertainty = None
        else:
            y_pred_out = y_pred_te
            uncertainty = None
    # Convert to array for downstream
    y_pred_te = np.array(y_pred_out if not isinstance(y_pred_out, dict) else y_pred_out.get("reg"))
    np.nan_to_num(y_pred_te, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if is_multitask and not enable_mh:
        # Extract regression predictions and inverse transform
        y_pred_reg = y_pred_te[:, 0]
        y_pred_cls = y_pred_te[:, 1]
        
        y_pred_reg = scalers.y_scaler.inverse_transform(y_pred_reg.reshape(-1,1)).reshape(-1)
        y_true_reg = y_te[:, 0]
        y_true_cls = y_te[:, 1]
        
        # Metrics for regression
        metrics = {
            "rmse": rmse(y_true_reg, y_pred_reg),
            "mae": mae(y_true_reg, y_pred_reg),
            "mape": mape(y_true_reg, y_pred_reg),
            "smape": smape(y_true_reg, y_pred_reg),
            "cls_accuracy": float(np.mean((y_pred_cls > 0.5) == y_true_cls)),
            "cls_precision": float(np.mean(y_true_cls[y_pred_cls > 0.5])) if np.sum(y_pred_cls > 0.5) > 0 else 0.0
        }
        
        result = (metrics, y_true_reg.tolist(), y_pred_reg.tolist(), y_true_cls.tolist(), y_pred_cls.tolist())
    elif is_multitask and enable_mh:
        # Multi-horizon outputs: y_pred_out is dict
        y_pred_reg = y_pred_out["reg"]  # [N,H]
        y_pred_cls = y_pred_out["cls"]  # [N,H]
        # Inverse scale regression per horizon using same scaler (approx)
        y_pred_reg_inv = scalers.y_scaler.inverse_transform(y_pred_reg.reshape(-1,1)).reshape(y_pred_reg.shape)
        # Compute multi-horizon ground truth from prices
        lookback = int(cfg.get("lookback", 60))
        test_dates = _load_dates_for_fold(ds_path, fold, lookback, cfg)
        mh_metrics = {}
        mh_uq_payload = None
        if test_dates is not None:
            y_true_mh = _compute_true_multihorizon_returns(ticker, test_dates, horizons)
            for i, H in enumerate(horizons):
                yt = y_true_mh.get(H)
                yp = y_pred_reg_inv[:, i]
                mask = np.isfinite(yt) & np.isfinite(yp)
                if mask.sum() > 0:
                    mh_metrics[f"da@{H}"] = float(directional_accuracy(yt[mask], yp[mask]))
                    mh_metrics[f"ic@{H}"] = float(information_coefficient(yt[mask], yp[mask]))
                    mh_metrics[f"sharpe_like@{H}"] = float(sharpe_like(yp[mask]))
                else:
                    mh_metrics[f"da@{H}"] = 0.0
                    mh_metrics[f"ic@{H}"] = 0.0
                    mh_metrics[f"sharpe_like@{H}"] = float(sharpe_like(yp))
        else:
            for i, H in enumerate(horizons):
                mh_metrics[f"sharpe_like@{H}"] = float(sharpe_like(y_pred_reg_inv[:, i]))
        if uncertainty is not None:
            mh_uq_payload = {
                "mean": y_pred_reg_inv.tolist(),
                "std": uncertainty.get("std").tolist() if isinstance(uncertainty.get("std"), np.ndarray) else None,
                "q10": uncertainty.get("q10").tolist() if isinstance(uncertainty.get("q10"), np.ndarray) else None,
                "q90": uncertainty.get("q90").tolist() if isinstance(uncertainty.get("q90"), np.ndarray) else None,
                "horizons": horizons
            }
        result = {"fold": fold, "mh_metrics": mh_metrics, "mh_uq": mh_uq_payload}
    else:
        y_pred_te = y_pred_te.reshape(-1)
        y_pred_te = scalers.y_scaler.inverse_transform(y_pred_te.reshape(-1,1)).reshape(-1)
        
        # Check if original targets were multi-task but model is single-task
        if len(y_te.shape) > 1 and y_te.shape[1] == 2:
            # Use only regression target for single-task model
            y_true_te = y_te[:, 0]
        else:
            y_true_te = y_te

        metrics = {
            "rmse": rmse(y_true_te, y_pred_te),
            "mae": mae(y_true_te, y_pred_te),
            "mape": mape(y_true_te, y_pred_te),
            "smape": smape(y_true_te, y_pred_te),
        }

        result = (metrics, y_true_te.tolist(), y_pred_te.tolist(), None, None)

    # Cleanup to free host memory after each fold
    try:
        del ds_tr, ds_va, ds_te
    except Exception:
        pass
    try:
        del X_tr, X_va, X_te, X_trs, X_vas, X_tes, y_tr, y_va, y_te, y_trs
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    tf.keras.backend.clear_session()
    gc.collect()

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ensemble", action="store_true", help="Run ensemble training with multiple seeds")
    ap.add_argument("--gpu", action="store_true", help="Enable GPU training optimizations")
    # New CLI overrides
    ap.add_argument("--horizons", type=str, default=None, help="Comma-separated horizons, e.g. 5,10,20")
    ap.add_argument("--loss_type", type=str, default=None, choices=["huber","pinball"], help="Loss type for regression")
    ap.add_argument("--lambda_cls", type=float, default=None, help="Lambda for classification loss weight")
    ap.add_argument("--n_mc_dropout", type=int, default=None, help="Number of MC Dropout passes at inference")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    # Apply CLI overrides
    if args.horizons is not None:
        cfg["horizons"] = [int(x) for x in args.horizons.split(",") if x.strip()]
    if args.loss_type is not None:
        cfg["loss_type"] = args.loss_type
    if args.lambda_cls is not None:
        cfg["lambda_cls"] = float(args.lambda_cls)
    if args.n_mc_dropout is not None:
        cfg["n_mc_dropout"] = int(args.n_mc_dropout)

    # Setup GPU if requested
    if args.gpu:
        setup_gpu()

    models = cfg["models"]
    lookback = int(cfg.get("lookback", 60))
    
    # Ensemble settings
    ensemble_config = cfg.get("ensemble", {})
    enable_ensemble = args.ensemble or ensemble_config.get("enable", False)
    seeds = ensemble_config.get("seeds", [42]) if enable_ensemble else [42]
    ensemble_lookbacks = cfg.get("ensemble_lookbacks", [lookback])
    
    for ticker in cfg["tickers"]:
        for lb in ensemble_lookbacks:
            ds_path = Path("datasets") / ticker / f"L{lb}"
            if not ds_path.exists():
                print(f"[WARN] Dataset not found: {ds_path}. Run prepare_dataset.py first.")
                continue
                
            folds = sorted(int(p.stem.replace("fold","")) for p in ds_path.glob("fold*.npz"))
            
            for model_name in models:
                if enable_ensemble:
                    # Ensemble training with multiple seeds
                    all_metrics = []
                    all_preds = []
                    
                    for seed in seeds:
                        print(f"Training {ticker} {model_name} L{lb} with seed {seed}...")
                        seed_everything(seed)
                        
                        out_dir = Path("reports") / ticker / model_name / f"L{lb}" / f"seed{seed}"
                        out_dir.mkdir(parents=True, exist_ok=True)

                        metrics_list = []
                        preds_all = []
                        
                        for f in folds:
                            result = train_one_fold(model_name, ds_path, f, cfg, seed, ticker)
                            if isinstance(result, dict) and "mh_metrics" in result:
                                metrics_list.append({"fold": f, "seed": seed, **result["mh_metrics"]})
                                if result.get("mh_uq"):
                                    save_json(out_dir / f"preds_with_uncertainty_fold{f}.json", result["mh_uq"])
                                continue
                            if len(result) == 5:  # Multi-task single horizon
                                metrics, y_true_reg, y_pred_reg, y_true_cls, y_pred_cls = result
                                metrics_list.append({"fold": f, "seed": seed, **metrics})
                                preds_all.append({
                                    "fold": f, "seed": seed,
                                    "y_true_reg": y_true_reg, "y_pred_reg": y_pred_reg,
                                    "y_true_cls": y_true_cls, "y_pred_cls": y_pred_cls
                                })
                            else:  # Single task
                                metrics, y_true, y_pred = result
                                metrics_list.append({"fold": f, "seed": seed, **metrics})
                                preds_all.append({"fold": f, "seed": seed, "y_true": y_true, "y_pred": y_pred})

                        save_json(out_dir / "metrics.json", metrics_list)
                        save_json(out_dir / "preds.json", preds_all)

                        # Aggregate UQ files if present
                        uq_files = sorted(out_dir.glob("preds_with_uncertainty_fold*.json"))
                        if uq_files:
                            import json
                            means = [] ; stds = [] ; q10s = [] ; q90s = [] ; horizons_ref = None
                            for p in uq_files:
                                d = json.load(open(p, "r", encoding="utf-8"))
                                means.append(np.array(d.get("mean")))
                                if d.get("std") is not None:
                                    stds.append(np.array(d.get("std")))
                                if d.get("q10") is not None:
                                    q10s.append(np.array(d.get("q10")))
                                if d.get("q90") is not None:
                                    q90s.append(np.array(d.get("q90")))
                                horizons_ref = d.get("horizons", horizons_ref)
                            agg = {
                                "mean": np.concatenate(means, axis=0).tolist() if means else None,
                                "std": (np.concatenate(stds, axis=0).tolist() if stds else None),
                                "q10": (np.concatenate(q10s, axis=0).tolist() if q10s else None),
                                "q90": (np.concatenate(q90s, axis=0).tolist() if q90s else None),
                                "horizons": horizons_ref
                            }
                            save_json(out_dir / "preds_with_uncertainty.json", agg)
                        
                        all_metrics.extend(metrics_list)
                        all_preds.extend(preds_all)
                    
                    # Save ensemble results
                    ensemble_dir = Path("reports") / ticker / model_name / f"L{lb}" / "ensemble"
                    ensemble_dir.mkdir(parents=True, exist_ok=True)
                    save_json(ensemble_dir / "all_metrics.json", all_metrics)
                    save_json(ensemble_dir / "all_preds.json", all_preds)
                    
                    # Calculate ensemble statistics
                    import pandas as pd
                    mdf = pd.DataFrame(all_metrics)
                    ensemble_stats = mdf.groupby('fold').agg({
                        'rmse': ['mean', 'std'],
                        'mae': ['mean', 'std'],
                        'mape': ['mean', 'std'],
                        'smape': ['mean', 'std']
                    }).round(6)
                    ensemble_stats.to_csv(ensemble_dir / "ensemble_stats.csv")
                    print(f"[OK] {ticker} {model_name} L{lb} ensemble stats:")
                    print(ensemble_stats)
                    
                else:
                    # Single seed training (original behavior)
                    out_dir = Path("reports") / ticker / model_name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    metrics_list = []
                    preds_all = []

                    for f in folds:
                        result = train_one_fold(model_name, ds_path, f, cfg, ticker=ticker)
                        if isinstance(result, dict) and "mh_metrics" in result:
                            metrics_list.append({"fold": f, **result["mh_metrics"]})
                            if result.get("mh_uq"):
                                save_json(out_dir / f"preds_with_uncertainty_fold{f}.json", result["mh_uq"])
                            continue
                        if len(result) == 5:  # Multi-task single horizon
                            metrics, y_true_reg, y_pred_reg, y_true_cls, y_pred_cls = result
                            metrics_list.append({"fold": f, **metrics})
                            preds_all.append({
                                "fold": f,
                                "y_true_reg": y_true_reg, "y_pred_reg": y_pred_reg,
                                "y_true_cls": y_true_cls, "y_pred_cls": y_pred_cls
                            })
                        else:  # Single task
                            metrics, y_true, y_pred = result
                            metrics_list.append({"fold": f, **metrics})
                            preds_all.append({"fold": f, "y_true": y_true, "y_pred": y_pred})

                    save_json(out_dir / "metrics.json", metrics_list)
                    save_json(out_dir / "preds.json", preds_all)

                    # Aggregate UQ files if present
                    uq_files = sorted(out_dir.glob("preds_with_uncertainty_fold*.json"))
                    if uq_files:
                        import json
                        means = [] ; stds = [] ; q10s = [] ; q90s = [] ; horizons_ref = None
                        for p in uq_files:
                            d = json.load(open(p, "r", encoding="utf-8"))
                            means.append(np.array(d.get("mean")))
                            if d.get("std") is not None:
                                stds.append(np.array(d.get("std")))
                            if d.get("q10") is not None:
                                q10s.append(np.array(d.get("q10")))
                            if d.get("q90") is not None:
                                q90s.append(np.array(d.get("q90")))
                            horizons_ref = d.get("horizons", horizons_ref)
                        agg = {
                            "mean": np.concatenate(means, axis=0).tolist() if means else None,
                            "std": (np.concatenate(stds, axis=0).tolist() if stds else None),
                            "q10": (np.concatenate(q10s, axis=0).tolist() if q10s else None),
                            "q90": (np.concatenate(q90s, axis=0).tolist() if q90s else None),
                            "horizons": horizons_ref
                        }
                        save_json(out_dir / "preds_with_uncertainty.json", agg)

                    import pandas as pd
                    mdf = pd.DataFrame(metrics_list)
                    mdf.to_csv(out_dir / "metrics_summary.csv", index=False)
                    print(f"[OK] {ticker} {model_name} metrics:\n", mdf.describe(include="all"))

if __name__ == "__main__":
    main()