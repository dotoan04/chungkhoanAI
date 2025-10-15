# ChungKhoanAI - Hệ Thống Dự Đoán Chứng Khoán Tiên Tiến

Hệ thống dự đoán chứng khoán Việt Nam sử dụng **Deep Learning** với kiến trúc **TCN-Residual** (Temporal Convolutional Network) và **Multi-Horizon Prediction** cho phân tích thị trường chứng khoán.

## 🏗️ Tổng Quan Kiến Trúc

### Mô Hình TCN-Residual
Kiến trúc TCN phức tạp với multi-task learning:

```
Input L×d (L=60, d=65 features)
 └─> Causal 1D Conv (k=5, f=64, dilation=1) + LayerNorm + ReLU + Dropout(0.1)
     Residual add
 └─> Causal 1D Conv (k=5, f=64, dilation=2) + LN + ReLU + Dropout(0.1) + Residual
 └─> Causal 1D Conv (k=5, f=64, dilation=4) + LN + ReLU + Dropout(0.1) + Residual
 └─> Causal 1D Conv (k=5, f=64, dilation=8) + LN + ReLU + Dropout(0.1) + Residual
 └─> Squeeze-Excitation (channel attention)
 └─> GlobalAvgPool (theo thời gian)
 └─> Multi-Horizon Heads:
     ├─> Regression: Dense(64) → Dense(3) → log-return @5,10,20 days
     └─> Classification: Dense(64) → Dense(3, sigmoid) → direction @5,10,20 days
```

### Tính Năng Chính

- **Multi-Horizon Prediction**: Dự đoán log-return và hướng tại 3 horizon (5, 10, 20 ngày)
- **MC-Dropout Uncertainty**: Ước lượng uncertainty với 20 forward passes
- **Advanced Loss Functions**: Huber loss hoặc Pinball loss với λ·BCE
- **Ensemble Strategy**: 5 seeds × 2 window sizes × multiple architectures
- **Walk-Forward Validation**: Xác thực chuỗi thời gian thực tế
- **Paper Trading**: Backtesting với uncertainty-aware entry rules

## 📋 Đặc Tả Mô Hình

### Tham Số Kiến Trúc
- **Độ dài Cửa sổ (L)**: 60 (chính), 90 (ensemble)
- **Kích thước Kernel (k)**: 5
- **Filters**: 64 (có thể mở rộng đến 96-128)
- **Dilations**: [1, 2, 4, 8]
- **MC-Dropout**: 0.1 ở output heads
- **Activation**: ReLU + LayerNorm

### Hàm Loss
```
L = Loss_reg + λ·Loss_cls
```
Trong đó:
- **Regression**: Huber loss (δ=1.0) hoặc Pinball loss (quantile 0.5)
- **Classification**: Binary Cross-Entropy cho direction prediction
- **λ=0.3** (hệ số trọng số configurable)

### Uncertainty Estimation
- **MC-Dropout**: 20 forward passes với training=True
- **Output**: Mean, Std, Q10, Q90 cho mỗi horizon
- **Entry Rule**: |mean_pred| > fee + buffer AND [Q10, Q90] không chứa 0

### Tối Ưu Hóa
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4, clipnorm=1.0)
- **Mixed Precision**: FP16 cho GPU optimization
- **Batch Size**: 64 (có thể tăng lên 256 cho GPU mạnh)
- **Early Stopping**: patience=15
- **Memory Management**: tf.data pipeline + cleanup per fold

## 🚀 Quy Trình Sử Dụng

### 1. Cài Đặt
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -U -r requirements.txt
```

### 2. Chuẩn Bị Dữ Liệu (Chạy 1 lần)
```bash
# Thu thập dữ liệu từ VnStock API
python src/collect_vnstock.py --tickers FPT HPG VNM --start 2015-01-01 --end 2025-08-28

# Chuẩn bị dataset với 65 features
python src/prepare_dataset.py --config configs/config.yaml
```

### 3. Training Multi-Horizon TCN
```bash
# Multi-horizon với MC-Dropout uncertainty
python src/train.py --config configs/config.yaml --gpu \
  --horizons 5,10,20 --loss_type huber --lambda_cls 0.3 --n_mc_dropout 20

# Hoặc chỉ TCN single-horizon (backward compatible)
python src/train.py --config configs/config_baseline.yaml --gpu
```

### 4. Backtesting với Uncertainty
```bash
# Backtest theo horizon H ∈ {5,10,20}
python src/paper_trading.py --config configs/config.yaml --ticker FPT --horizon 10 --model tcn
python src/paper_trading.py --config configs/config.yaml --ticker FPT --horizon 20 --model tcn
```

### 5. Trực Quan Hóa Kết Quả
Sử dụng notebook cell đã cung cấp để plot:
- Time-series True vs Predicted cho từng fold
- Scatter plot với RMSE/MAE/SMAPE
- Multi-horizon metrics: DA@H, IC@H, Sharpe-like@H

## 📁 Cấu Trúc Dự Án

```
ChungKhoanAI/
├── src/
│   ├── models.py          # TCN-Residual + Multi-horizon heads
│   ├── train.py           # Training với MC-Dropout + CLI overrides
│   ├── paper_trading.py   # Uncertainty-aware backtesting
│   ├── prepare_dataset.py # 65 technical features + multi-task targets
│   ├── ensemble.py        # Ensemble prediction
│   ├── backtest.py        # Traditional backtesting
│   ├── features.py        # Technical indicators (robust BBands)
│   ├── evaluate.py        # DA, IC, Sharpe-like metrics
│   ├── collect_vnstock.py # VnStock data collection
│   └── utils.py           # Utilities + walk-forward splits
├── configs/
│   ├── config.yaml        # Multi-horizon configuration
│   └── config_baseline.yaml # Single-horizon baseline
├── reports/               # Results organized by ticker/model
│   ├── {TICKER}/
│   │   ├── tcn/
│   │   │   ├── L60/
│   │   │   │   ├── seed{X}/ # Individual seed results
│   │   │   │   │   ├── metrics.json # DA@5, IC@5, sharpe_like@5, etc.
│   │   │   │   │   └── preds_with_uncertainty.json # Mean, std, q10, q90
│   │   │   │   └── ensemble/ # Ensemble results
│   │   │   └── L90/
│   │   ├── gru/           # Comparison models
│   │   └── lstm/
│   └── ...
├── datasets/              # Preprocessed time series
├── data/                  # Raw price data
└── requirements.txt       # Dependencies
```

## ⚙️ Cấu Hình Chi Tiết

### Multi-Horizon Config (`configs/config.yaml`)
```yaml
# Core settings
tickers: ["FPT", "HPG", "VNM"]
models: ["tcn", "gru", "lstm"]
lookback: 60
batch_size: 64  # Increase to 256 for stronger GPUs

# Multi-horizon prediction
horizons: [5, 10, 20]
loss_type: huber        # huber | pinball
lambda_cls: 0.3         # Classification loss weight
n_mc_dropout: 20        # MC-Dropout passes

# Trading parameters
trade_fee: 0.002        # 0.2% trading fee
risk_per_trade: 0.01    # 1% risk per trade

# TCN architecture
tcn:
  filters: 64
  kernel_size: 5
  dilations: [1, 2, 4, 8]
  dropout_rate: 0.1
  use_se: true              # Squeeze-Excitation
  use_multitask: true       # Multi-task learning
  weight_decay: 1e-4

# Ensemble settings
ensemble:
  enable: true
  seeds: [42, 123, 456, 789, 999]  # 5 different seeds
```

### CLI Overrides
```bash
# Override any config parameter via command line
python src/train.py --config configs/config.yaml --gpu \
  --horizons 5,10,20 \
  --loss_type pinball \
  --lambda_cls 0.5 \
  --n_mc_dropout 30
```

## 📊 Kết Quả và Metrics

### Multi-Horizon Metrics
- **DA@H**: Directional Accuracy tại horizon H
- **IC@H**: Information Coefficient (correlation) tại horizon H  
- **Sharpe-like@H**: Mean(pred) / Std(pred) tại horizon H
- **Traditional**: RMSE, MAE, MAPE, SMAPE (backward compatible)

### Uncertainty Outputs
```json
{
  "mean": [[r5, r10, r20], ...],     // Mean predictions per horizon
  "std": [[σ5, σ10, σ20], ...],      // Standard deviation
  "q10": [[q10_5, q10_10, q10_20], ...], // 10th percentile
  "q90": [[q90_5, q90_10, q90_20], ...], // 90th percentile
  "horizons": [5, 10, 20]
}
```

### Backtesting Results
```json
{
  "equity_curve": [1.0, 1.05, 0.98, ...],
  "max_drawdown": 0.15,
  "winrate": 0.62,
  "trades": 145
}
```

## 🔬 Tính Năng Tiên Tiến

### MC-Dropout Uncertainty
- **Training**: Dropout ở output heads (rate=0.1)
- **Inference**: 20 forward passes với training=True
- **Aggregation**: Mean, std, quantiles cho confidence intervals
- **Trading**: Entry chỉ khi [Q10, Q90] không chứa 0

### Multi-Horizon Learning
- **Targets**: Log-returns tại H=5,10,20 ngày
- **Architecture**: Shared encoder + separate heads per horizon
- **Loss**: Combined regression + classification loss
- **Evaluation**: Horizon-specific metrics

### Advanced Loss Functions
- **Huber Loss**: Robust to outliers (δ=1.0)
- **Pinball Loss**: Quantile regression (q=0.5)
- **Combined**: L_reg + λ·L_cls với configurable λ

### GPU Optimization
- **Mixed Precision**: FP16 training
- **tf.data Pipeline**: Efficient data loading
- **Memory Management**: Cleanup per fold
- **Batch Size**: Auto-scaling based on GPU memory

## 🎯 Chiến Lược Trading

### Entry Rules (Uncertainty-Aware)
1. **Signal Strength**: |mean_pred| > fee + buffer
2. **Confidence**: [Q10, Q90] không chứa 0 (no uncertainty overlap)
3. **Risk Management**: Fixed risk per trade (1% default)

### Exit Rules
- **Time-based**: Đóng sau đúng H ngày
- **Stop Loss**: 2×ATR từ entry
- **Take Profit**: 4×ATR từ entry

### Portfolio Management
- **Position Sizing**: Risk-based sizing
- **Diversification**: Multiple horizons, multiple tickers
- **Transaction Costs**: Realistic fee modeling (0.2%)

## 📈 Hiệu Suất Kỳ Vọng

### Multi-Horizon Performance
- **DA@5**: 55-65% (directional accuracy)
- **DA@10**: 52-62%
- **DA@20**: 50-60%
- **IC@5**: 0.05-0.15 (information coefficient)
- **Sharpe-like**: 5-20 (depends on volatility)

### Trading Performance
- **Win Rate**: 55-65%
- **Max Drawdown**: 10-25%
- **Sharpe Ratio**: 0.8-1.5
- **Annual Return**: 15-30% (before costs)

### Ensemble Benefits
- **Variance Reduction**: 10-20% improvement
- **Robustness**: Better out-of-sample performance
- **Uncertainty**: More reliable confidence intervals

## 🛠️ Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch_size hoặc tắt ensemble
2. **NaN Metrics**: Đã fix với safe metrics computation
3. **Data Loading**: Check pandas_ta availability
4. **Syntax Errors**: All files đã compile OK

### Performance Tuning
```bash
# Increase GPU utilization
# Edit config.yaml:
batch_size: 256          # Increase for more VRAM usage
tcn:
  filters: 96           # Larger model capacity
  dilations: [1,2,4,8,16] # More layers

# Reduce memory usage
n_mc_dropout: 10        # Fewer MC passes
ensemble:
  enable: false         # Disable ensemble
```

## 🧪 Testing và Validation

### Quick Test
```bash
# Test single ticker, single horizon
python src/train.py --config configs/config.yaml --gpu \
  --horizons 5 --n_mc_dropout 5

# Test backtesting
python src/paper_trading.py --config configs/config.yaml \
  --ticker FPT --horizon 5 --model tcn
```

### Full Pipeline
```bash
# Complete end-to-end test
python src/run_all.py  # If available
```

## 📚 Nền Tảng Nghiên Cứu

Triển khai này dựa trên:
- **Temporal Convolutional Networks** cho sequence modeling
- **Multi-Task Learning** cho joint regression + classification
- **MC-Dropout** cho Bayesian uncertainty estimation
- **Ensemble Methods** cho variance reduction
- **Walk-Forward Validation** cho realistic evaluation

## 🤝 Mở Rộng

### Thêm Mô Hình Mới
1. Implement trong `src/models.py`
2. Update `configs/config.yaml`
3. Test với multi-horizon pipeline

### Thêm Metrics Mới
1. Extend `src/evaluate.py`
2. Update training loop trong `src/train.py`
3. Log vào `metrics.json`

### Thêm Trading Strategies
1. Extend `src/paper_trading.py`
2. Add new entry/exit rules
3. Implement portfolio optimization

---

**Lưu ý**: Hệ thống này được thiết kế cho mục đích nghiên cứu và giáo dục. Luôn xác thực kết quả và cân nhắc rủi ro thị trường trước khi đưa ra quyết định giao dịch thực tế.