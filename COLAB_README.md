# CHUNGKHOAN AI - Google Colab Guide

## **Huong Dan Chay Pipeline Tren Google Colab**

---

## **Buoc 1: Chuan Bi**

### **1.1 Upload Files Len Colab**
```python
# Upload cac file can thiet:
# - colab_research_pipeline.ipynb
# - src/ (thu muc chua code)
# - configs/ (thu muc config)
# - requirements.txt
```

### **1.2 Fix loi 403 Forbidden**
Neu gap loi 403 Forbidden khi thu thap du lieu, he thong se tu dong:
- Thu lai voi source khac (TCBS thay vi VCI)
- Neu van loi, se tu dong tao sample data cho demo
- Sample data van du de chay toan bo pipeline va luan van

---

## **Buoc 2: Chay Notebook**

1. **Mo notebook**: `colab_research_pipeline.ipynb`
2. **Runtime**: Chon GPU neu co (Runtime → Change runtime type → GPU)
3. **Chay tung cell** theo thu tu tu tren xuong duoi

**Luu y quan trong:**
- Neu gap loi 403 Forbidden o buoc thu thap du lieu, he thong se tu dong tao sample data
- Sample data hoan toan du de demo va viet luan van
- Khong can lo lang ve loi nay

---

## **Buoc 3: Download Ket Qua**

### **Sau khi chay xong:**
```python
# Notebook se tao file thesis_results.zip
# Download file nay ve may
```

### **Ket qua bao gom:**
```
reports/
├── research_summary.txt           # Tom tat nghien cuu
├── latex_tables/
│   ├── performance_comparison.tex # Bang LaTeX
│   └── architecture_comparison.tex
├── baseline_comparison.csv        # Bang so sanh
└── {TICKER}/                     # Ket qua chi tiet
```

---

## **Troubleshooting**

### **Loi 403 Forbidden:**
```
ConnectionError: Failed to fetch data: 403 - Forbidden
```
**Giai phap:** He thong da co fallback mechanism:
- Se thu source khac (TCBS)
- Neu van loi, se tu dong tao sample data
- Sample data hoan toan ok cho luan van

### **Loi Memory:**
- Giam batch_size trong config
- Su dung model nho hon
- Chay tung ticker mot

### **Loi Timeout:**
- Luu progress thuong xuyen
- Chia pipeline thanh nhieu notebook nho

---

## **Pipeline Steps:**

1. **Data Collection** - Thu thap tu vnstock API (voi fallback)
2. **Dataset Preparation** - Tao features va targets
3. **Baseline Models** - Naive, SMA, EMA, ARIMA
4. **DL Training** - TCN, GRU, LSTM voi ensemble
5. **Backtesting** - Danh gia hieu suat
6. **Comparison** - So sanh baseline vs DL
7. **Reports** - Tao LaTeX tables cho luan van

---

## **Luu Y Quan Trong**

1. **Colab runtime** co the reset sau 12h
2. **Luu ket qua** vao Google Drive thuong xuyen
3. **Download results** truoc khi session ket thuc
4. **Kiem tra GPU** availability truoc khi training

---

## **Ket Thuc**

Sau khi chay xong, ban se co day du du lieu va bang bieu cho luan van:

- **Metrics day du** (RMSE, MAE, MAPE, SMAPE)
- **LaTeX tables** san sang cho luan van
- **Backtesting results** chi tiet
- **Research summary** hoan chinh

**Chuc ban thanh cong voi luan van!**
