#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, yaml
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import gc

# GPU Configuration
def setup_gpu(cfg):
    """Cấu hình GPU tối ưu cho T4 15GB VRAM"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set memory limit from config (default 14GB for T4)
            memory_limit = cfg.get("gpu", {}).get("memory_limit", 14336)  # 14GB in MB
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )

            print(f"🚀 Using GPU: {gpus[0].name}")
            print(f"🔧 Memory limit set to {memory_limit}MB")

            # Mixed precision if enabled in config
            if cfg.get("gpu", {}).get("mixed_precision", True):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("🔥 Mixed precision FP16 enabled")

            return True

        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            print("Falling back to CPU...")
            return False
    else:
        print("⚠️ No GPU found, using CPU")
        return False

from utils import seed_everything, fit_transform_scalers, apply_scalers, save_json
from models import make_model, WarmupCosineDecay
from evaluate import rmse, mae, mape, smape

def train_one_fold(model_name, ds_path: Path, fold: int, cfg, seed: int = 42):
    # Set seed for reproducibility
    seed_everything(seed)
    
    npz = np.load(ds_path / f"fold{fold}.npz")
    X_tr, y_tr = npz["X_tr"], npz["y_tr"]
    X_va, y_va = npz["X_va"], npz["y_va"]
    X_te, y_te = npz["X_te"], npz["y_te"]
    
    # Check if this is multi-task learning (y has 2 dimensions AND model supports multi-task)
    is_multitask = (len(y_tr.shape) > 1 and y_tr.shape[1] == 2 and model_name.lower() == "tcn")
    
    if is_multitask:
        # Split targets: regression and classification  
        y_tr_reg, y_tr_cls = y_tr[:, 0], y_tr[:, 1]
        y_va_reg, y_va_cls = y_va[:, 0], y_va[:, 1]
        y_te_reg, y_te_cls = y_te[:, 0], y_te[:, 1]
        
        # Scale only regression targets, keep classification as is
        scalers, X_trs, y_trs_reg = fit_transform_scalers(X_tr, y_tr_reg)
        X_vas, y_vas_reg = apply_scalers(scalers, X_va, y_va_reg)
        X_tes, y_tes_reg = apply_scalers(scalers, X_te, y_te_reg)
        
        # Combine scaled regression + unscaled classification targets
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
    
    # Ensure X data is float32
    X_trs = X_trs.astype(np.float32)
    X_vas = X_vas.astype(np.float32)
    X_tes = X_tes.astype(np.float32)

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
            "use_multitask": is_multitask
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
        loss_lambda = float(tcn_config.get("loss_lambda", 0.25))
        
        # Simple recompile with AdamW and better settings
        if is_multitask:
            from models import huber_bce_loss
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=lr, 
                    weight_decay=weight_decay,
                    clipnorm=1.0
                ),
                loss=huber_bce_loss(delta=1.0, lam=loss_lambda),
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
    
    # tf.data input pipeline + GPU optimizations
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU training with optimized batch_size={batch_size} for T4 15GB")

    train_ds = tf.data.Dataset.from_tensor_slices((X_trs, y_trs)).shuffle(len(X_trs)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_vas, y_vas)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0,
        callbacks=cbs
    )

    # Predictions (batched)
    test_ds = tf.data.Dataset.from_tensor_slices(X_tes).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    y_pred_te = model.predict(test_ds, verbose=0)
    
    if is_multitask:
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
        
        # Clear memory before returning
        del model, X_trs, X_vas, X_tes, y_trs, y_vas, y_tes
        gc.collect()
        tf.keras.backend.clear_session()

        return metrics, y_true_reg.tolist(), y_pred_reg.tolist(), y_true_cls.tolist(), y_pred_cls.tolist()
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

        # Clear memory before returning
        del model, X_trs, X_vas, X_tes, y_trs, y_vas, y_tes
        gc.collect()
        tf.keras.backend.clear_session()

        return metrics, y_true_te.tolist(), y_pred_te.tolist(), None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ensemble", action="store_true", help="Run ensemble training with multiple seeds")
    ap.add_argument("--no-gpu", action="store_true", help="Disable GPU optimizations")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # Setup GPU (unless disabled)
    if not args.no_gpu:
        gpu_enabled = setup_gpu(cfg)
        if not gpu_enabled:
            print("⚠️ GPU setup failed, using CPU")
    else:
        print("⚠️ GPU disabled by user, using CPU")

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
                            result = train_one_fold(model_name, ds_path, f, cfg, seed)
                            
                            if len(result) == 5:  # Multi-task
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
                        result = train_one_fold(model_name, ds_path, f, cfg)
                        
                        if len(result) == 5:  # Multi-task
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

                    import pandas as pd
                    mdf = pd.DataFrame(metrics_list)
                    mdf.to_csv(out_dir / "metrics_summary.csv", index=False)
                    print(f"[OK] {ticker} {model_name} metrics:\n", mdf.describe(include="all"))

if __name__ == "__main__":
    main()