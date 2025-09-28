import argparse, json, yaml
from pathlib import Path
import numpy as np
import pandas as pd

def load_config(path):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def should_trade(mean_pred, q10, q90, fee, buffer=0.0):
    return (abs(mean_pred) > (fee + buffer)) and not (q10 <= 0.0 <= q90)

def backtest_horizon(prices: pd.Series, preds_mean: np.ndarray, q10: np.ndarray, q90: np.ndarray,
                     atr: pd.Series, fee: float, risk_per_trade: float, H: int):
    equity = [1.0]
    peak = 1.0
    dd = [0.0]
    wins = 0
    trades = 0

    i = 0
    while i < len(prices) - H:
        m = preds_mean[i]
        if not should_trade(m, q10[i], q90[i], fee, buffer=0.0):
            equity.append(equity[-1])
            peak = max(peak, equity[-1])
            dd.append(peak - equity[-1])
            i += 1
            continue

        # Position sizing (fixed risk)
        atr_i = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else float(np.nanmean(atr.iloc[max(0,i-20):i+1]))
        if atr_i <= 0 or not np.isfinite(atr_i):
            atr_i = 0.01 * prices.iloc[i]
        stop = 2.0 * atr_i
        take = 4.0 * atr_i
        direction = 1.0 if m > 0 else -1.0

        entry = prices.iloc[i]
        exit_price = prices.iloc[i+H]

        pnl = direction * (exit_price - entry) - fee * entry
        trades += 1
        if pnl > 0:
            wins += 1

        eq = equity[-1] * (1.0 + (pnl / entry) * risk_per_trade)
        equity.append(eq)
        peak = max(peak, eq)
        dd.append(peak - eq)

        i += H

    return {
        "equity_curve": equity,
        "max_drawdown": float(max(dd) if dd else 0.0),
        "winrate": float(wins / trades) if trades > 0 else 0.0,
        "trades": trades
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ticker", type=str, required=True)
    ap.add_argument("--horizon", type=int, choices=[5,10,20], required=True)
    ap.add_argument("--model", type=str, default="tcn")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fee = float(cfg.get("trade_fee", 0.002))
    risk = float(cfg.get("risk_per_trade", 0.01))

    # Load predictions with uncertainty
    rep_dir = Path("reports") / args.ticker / args.model / f"L{cfg.get('lookback',60)}"
    ens_dir = rep_dir / "ensemble"
    preds_uq = rep_dir / "preds_with_uncertainty.json"
    if ens_dir.exists():
        preds_uq = ens_dir / "preds_with_uncertainty.json"
    if not preds_uq.exists():
        print(f"[WARN] No preds_with_uncertainty.json found at {preds_uq}")
        return

    data = json.load(open(preds_uq, "r", encoding="utf-8"))
    # Expect keys: mean, std, q10, q90 per horizon index
    idx = {5:0, 10:1, 20:2}[args.horizon]
    mean = np.array(data["mean"])[:, idx]
    q10 = np.array(data["q10"])[:, idx]
    q90 = np.array(data["q90"])[:, idx]

    # Load price and ATR (for SL/TP sizing)
    prices = pd.read_csv(Path("data")/args.ticker/"prices_daily.csv")
    prices["date"] = pd.to_datetime(prices["time"] if "time" in prices.columns else prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)
    close = prices["close"].astype(float)
    # Simple ATR proxy
    atr = (prices["high"].astype(float) - prices["low"].astype(float)).rolling(14).mean().fillna(method='bfill')

    bt = backtest_horizon(close.iloc[-len(mean):].reset_index(drop=True), mean, q10, q90, atr.iloc[-len(mean):].reset_index(drop=True), fee, risk, args.horizon)
    out_dir = Path("reports") / args.ticker / args.model / f"L{cfg.get('lookback',60)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"backtest_H{args.horizon}.json", "w", encoding="utf-8") as f:
        json.dump(bt, f, ensure_ascii=False, indent=2)
    print(f"[OK] Backtest H={args.horizon}: winrate={bt['winrate']:.2%}, maxDD={bt['max_drawdown']:.4f}, trades={bt['trades']}")

if __name__ == "__main__":
    main()


