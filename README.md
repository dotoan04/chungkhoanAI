# ChungKhoanAI - Hệ Thống Dự Đoán Chứng Khoán TCN-Residual

Hệ thống dự đoán chứng khoán tiên tiến sử dụng kiến trúc **TCN-Residual (Temporal Convolutional Network với Residual Connections)** cho phân tích thị trường chứng khoán Việt Nam.

## 🏗️ Tổng Quan Kiến Trúc

### Mô Hình TCN-Residual
Mô hình cốt lõi triển khai kiến trúc TCN phức tạp với:

```
Input L×d (L=60, d=features)
 └─> Causal 1D Conv (k=5, f=64, dilation=1) + LayerNorm + ReLU + Dropout(0.1)
     Residual add
 └─> Causal 1D Conv (k=5, f=64, dilation=2) + LN + ReLU + Dropout(0.1) + Residual
 └─> Causal 1D Conv (k=5, f=64, dilation=4) + LN + ReLU + Dropout(0.1) + Residual
 └─> Causal 1D Conv (k=5, f=64, dilation=8) + LN + ReLU + Dropout(0.1) + Residual
 └─> (Tùy chọn) Squeeze-Excitation (channel attention)
 └─> GlobalAvgPool (theo thời gian)
 └─> Head Hồi quy: Dense(64) → Dense(1)  =>  ŷ_reg = r̂_{t+1}
 └─> (Tùy chọn) Head Phân loại: Dense(64) → Dense(1, sigmoid) =>  p̂(up)
```

### Tính Năng Chính

- **Multi-Task Learning**: Học đồng thời hồi quy (dự đoán giá) và phân loại (dự đoán hướng)
- **Hàm Loss Tiên tiến**: Kết hợp Huber loss + λ·BCE loss
- **Chiến lược Ensemble**: 5 seeds × 2 kích thước cửa sổ × nhiều kiến trúc
- **Tối ưu hóa Phức tạp**: AdamW với cosine decay + warmup scheduling
- **Walk-Forward Cross-Validation**: Xác thực chuỗi thời gian thực tế

## 📋 Đặc Tả Mô Hình

### Tham Số Kiến Trúc
- **Độ dài Cửa sổ (L)**: 60 (chính), 90 (ensemble)
- **Kích thước Kernel (k)**: 5
- **Filters**: 64 (có thể mở rộng đến 96-128)
- **Dilations**: [1, 2, 4, 8, (16)]
- **Chuẩn hóa**: LayerNorm
- **Dropout**: 0.1-0.2
- **Activation**: ReLU

### Hàm Loss
```
L = Huber(r, r̂) + λ·BCE(y, ŷ)
```
Trong đó:
- Huber loss (δ=1.0) cho hồi quy
- Binary Cross-Entropy cho dự đoán hướng
- λ=0.25 (hệ số trọng số)

### Tối Ưu Hóa
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Schedule**: Cosine decay + 5 warmup epochs
- **Gradient Clipping**: norm=1.0
- **Batch Size**: 64
- **Early Stopping**: patience=15

### Chia Dữ Liệu
- **Train**: 60%
- **Validation**: 20%
- **Test**: 20%
- **Phương pháp**: Walk-Forward CV (expanding window)

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -U -r requirements.txt
```

### 2. Pipeline Hoàn Chỉnh
```bash
python src/run_all.py
```

Thao tác này chạy toàn bộ pipeline:
1. Thu thập dữ liệu từ VnStock API
2. Kỹ thuật tính năng và chuẩn bị dataset
3. Training ensemble đa seeds
4. Tạo dự đoán ensemble
5. Backtesting (mô hình riêng lẻ + ensemble)

### 3. Các Bước Thủ Công

#### Thu Thập Dữ Liệu
```bash
python src/collect_vnstock.py --tickers FPT HPG VNM --start 2015-01-01 --end 2025-08-28
```

#### Chuẩn Bị Dataset (Multi-task)
```bash
python src/prepare_dataset.py --config configs/config.yaml --multitask
```

#### Training với Ensemble
```bash
python src/train.py --config configs/config.yaml --ensemble
```

#### Tạo Dự Đoán Ensemble
```bash
python src/ensemble.py --config configs/config.yaml
```

#### Baseline Models (Naive, SMA/EMA, ARIMA)
```bash
# Run baseline models
python run_baselines.py --tickers FPT HPG VNM

# Compare with DL models (predicting prices instead of returns)
python src/prepare_dataset.py --config configs/config_baseline.yaml
python src/train.py --config configs/config_baseline.yaml
python compare_baselines.py
```

#### Backtesting
```bash
# Mô hình riêng lẻ
python src/backtest.py --config configs/config.yaml

# Ensemble
python src/backtest.py --config configs/config.yaml --ensemble
```

## 📁 Cấu Trúc Dự Án

```
ChungKhoanAI/
├── src/
│   ├── models.py          # TCN-Residual + các kiến trúc khác
│   ├── train.py           # Training multi-task với ensemble
│   ├── ensemble.py        # Tạo dự đoán ensemble
│   ├── prepare_dataset.py # Kỹ thuật tính năng + multi-task targets
│   ├── backtest.py        # Backtesting chiến lược
│   ├── baseline.py        # Baseline models (Naive, SMA/EMA, ARIMA)
│   ├── collect_vnstock.py # Thu thập dữ liệu
│   ├── features.py        # Chỉ báo kỹ thuật
│   ├── evaluate.py        # Metrics
│   ├── utils.py           # Tiện ích
│   └── run_all.py         # Pipeline hoàn chỉnh
├── configs/
│   └── config.yaml        # Cấu hình với tham số TCN
├── reports/               # Kết quả sắp xếp theo ticker/model
│   ├── {TICKER}/
│   │   ├── tcn/           # Kết quả mô hình TCN
│   │   │   ├── L60/       # Kích thước cửa sổ 60
│   │   │   │   ├── seed{X}/ # Kết quả seed riêng lẻ
│   │   │   │   └── ensemble/ # Kết quả ensemble
│   │   │   └── L90/       # Kích thước cửa sổ 90
│   │   ├── gru/           # So sánh GRU
│   │   ├── baseline/      # Baseline models (Naive, SMA/EMA, ARIMA)
│   │   └── ensemble/      # Ensemble đa mô hình
│   └── ...
├── datasets/              # Dữ liệu chuỗi thời gian đã chuẩn bị
├── data/                  # Dữ liệu giá thô
├── run_baselines.py      # Script chạy baseline models
├── test_tcn.py           # Kiểm tra validation
└── requirements.txt       # Dependencies
```

## ⚙️ Cấu Hình

File `configs/config.yaml` chứa tất cả tham số:

```yaml
tickers: ["FPT", "HPG", "VNM"]
models: ["tcn", "gru", "lstm"]  # TCN là mô hình chính

# Tham số riêng cho TCN
tcn:
  filters: 64
  kernel_size: 5
  dilations: [1, 2, 4, 8]
  dropout_rate: 0.1
  use_se: true              # Squeeze-Excitation
  use_multitask: true       # Multi-task learning
  loss_lambda: 0.25         # λ cho BCE trong combined loss
  weight_decay: 1e-4

# Cài đặt ensemble
ensemble:
  enable: true
  seeds: [42, 123, 456, 789, 999]  # 5 seeds khác nhau
  voting_method: "average"

ensemble_lookbacks: [60, 90]  # Nhiều kích thước cửa sổ
```

## 📊 Cấu Trúc Kết Quả

Kết quả được sắp xếp theo:
- **Ticker** (FPT, HPG, VNM)
- **Mô hình** (tcn, gru, lstm)
- **Kích thước Cửa sổ** (L60, L90)
- **Seed** (seed42, seed123, v.v.)
- **Ensemble** (dự đoán kết hợp)

Mỗi thư mục kết quả chứa:
- `metrics.json`: Metrics hiệu suất
- `preds.json`: Dự đoán (hồi quy + phân loại)
- `backtest_summary.json`: Hiệu suất giao dịch
- `backtest_curve.csv`: Mô phỏng giao dịch chi tiết
- `backtest_equity.png`: Trực quan hóa đường cong vốn

## 🔬 Multi-Task Learning

Hệ thống dự đoán đồng thời:
1. **Hồi quy**: Log return ngày tiếp theo (liên tục)
2. **Phân loại**: Hướng giá (tăng/giảm)

Lợi ích:
- Dự đoán hướng ổn định hơn
- Quản lý rủi ro tốt hơn
- Tín hiệu giao dịch cải thiện

## 🎯 Chiến Lược Ensemble

Ensemble kết hợp:
- **5 seeds** để giảm phương sai
- **2 kích thước cửa sổ** (60, 90) cho những góc nhìn khác nhau
- **Nhiều kiến trúc** (TCN + GRU) cho sự đa dạng

**Hiệu suất Kỳ vọng**: Giảm MAPE 5-15% so với mô hình đơn lẻ

## 📈 Backtesting

Hệ thống cung cấp backtesting toàn diện:
- **Metrics**: Tỷ lệ Sharpe, max drawdown, tỷ lệ thắng
- **Mô phỏng Giao dịch**: Chi phí giao dịch, thực thi thực tế
- **Multi-Task**: Đánh giá riêng biệt cho tín hiệu hồi quy vs phân loại
- **Trực quan hóa**: Đường cong vốn và so sánh hiệu suất

## 🛠️ Tính Năng Tiên Tiến

### Lên Lịch Learning Rate
- **Warmup**: Tăng tuyến tính trong 5 epochs
- **Cosine Decay**: Giảm mượt mà về gần zero
- **Gradient Clipping**: Ngăn chặn gradient exploding

### Regularization
- **Weight Decay**: L2 regularization qua AdamW
- **Dropout**: Regularization ngẫu nhiên
- **Layer Normalization**: Training ổn định

### Kỹ Thuật Tính Năng
- **Chỉ Báo Kỹ Thuật**: 50+ chỉ báo
- **Bối Cảnh Thị Trường**: Tương quan chỉ số, beta
- **Phát Hiện Chế Độ**: Chế độ biến động và xu hướng
- **Tính Năng Lịch**: Ngày trong tuần, tính mùa vụ

## 🧪 Kiểm Tra

Chạy kiểm tra validation:
```bash
python test_tcn.py
```

Hoặc validation đơn giản:
```bash
python simple_test.py
```

## 📚 Nền Tảng Nghiên Cứu

Triển khai này dựa trên:
- **Temporal Convolutional Networks** cho mô hình hóa chuỗi
- **Residual Connections** cho ổn định training
- **Multi-Task Learning** cho dự đoán mạnh mẽ
- **Ensemble Methods** cho giảm phương sai
- **Walk-Forward Validation** cho đánh giá thực tế

## 🔄 Ví Dụ Sử Dụng

### Chỉ Training TCN
```bash
python src/train.py --config configs/config.yaml
```

### Cấu Hình TCN Tùy Chỉnh
Sửa đổi `configs/config.yaml`:
```yaml
tcn:
  filters: 96           # Tăng sức chứa
  dilations: [1,2,4,8,16] # Thêm layers
  dropout_rate: 0.15    # Regularization nhiều hơn
```

### Ensemble với Mô hình Tùy chỉnh
```bash
python src/ensemble.py --config configs/config.yaml --models tcn gru
```

## 🎯 Hiệu Suất Kỳ Vọng

**Metrics Baseline** (phạm vi thường thấy):
- **RMSE**: 0.015-0.025
- **MAPE**: 8-15%
- **Tỷ lệ Sharpe**: 0.5-1.2
- **Max Drawdown**: 10-25%
- **Độ chính xác Phân loại**: 52-58%

**Cải thiện Ensemble**: Tốt hơn 5-15% so với mô hình đơn lẻ

## 🤝 Đóng Góp

Để mở rộng hệ thống:
1. Thêm mô hình mới trong `src/models.py`
2. Cập nhật cấu hình trong `configs/config.yaml`
3. Mở rộng logic ensemble trong `src/ensemble.py`
4. Thêm tests trong `test_tcn.py`

---

**Lưu ý**: Hệ thống này được thiết kế cho mục đích nghiên cứu và giáo dục. Luôn xác thực kết quả và cân nhắc rủi ro thị trường trước khi đưa ra quyết định giao dịch.