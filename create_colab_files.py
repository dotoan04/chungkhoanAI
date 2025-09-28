#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Google Colab files without Unicode issues
"""

import json
from pathlib import Path

def create_colab_notebook():
    """Create Google Colab notebook"""

    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CHUNGKHOAN AI - Google Colab Research Pipeline\n",
                    "\n",
                    "## **Luan van nghien cuu: Du doan gia co phieu su dung Deep Learning**\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## Tong quan\n",
                    "Notebook nay chay pipeline hoan chinh de tao tat ca ket qua can thiet cho luan van:\n",
                    "- Thu thap du lieu tu vnstock API\n",
                    "- Baseline models (Naive, SMA, EMA, ARIMA)\n",
                    "- Deep Learning models (TCN-Residual, GRU, LSTM)\n",
                    "- Ensemble learning\n",
                    "- Backtesting va danh gia\n",
                    "- Tao bao cao LaTeX cho luan van\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 1: Cai dat moi truong"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Kiem tra GPU\n",
                    "!nvidia-smi\n",
                    "\n",
                    "# Cai dat packages can thiet\n",
                    "!pip install -q vnstock pandas numpy pandas-ta scikit-learn matplotlib tensorflow statsmodels PyYAML tqdm scipy\n",
                    "\n",
                    "# Clone repository (neu can)\n",
                    "# !git clone https://github.com/your-repo/chungkhoan-ai.git\n",
                    "# %cd chungkhoan-ai\n",
                    "\n",
                    "print(\"Environment setup completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 2: Thu thap du lieu"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mount Google Drive (tuy chon)\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "\n",
                    "# Thay doi working directory\n",
                    "%cd /content/drive/MyDrive/ChungKhoanAI  # Thay doi duong dan phu hop\n",
                    "\n",
                    "# Thu thap du lieu\n",
                    "!python src/collect_vnstock.py --tickers FPT HPG VNM VNINDEX --start 2015-01-01 --end 2025-08-28\n",
                    "\n",
                    "print(\"Data collection completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 3: Chuan bi dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Chuan bi dataset cho baseline comparison\n",
                    "!python src/prepare_dataset.py --config configs/config_baseline.yaml\n",
                    "\n",
                    "print(\"Dataset preparation completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 4: Baseline Models"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Chay baseline models\n",
                    "!python run_baselines.py --tickers FPT HPG VNM\n",
                    "\n",
                    "print(\"Baseline models completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 5: Deep Learning Models"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Training TCN models (co the mat 10-15 phut)\n",
                    "!python src/train.py --config configs/config_baseline.yaml\n",
                    "\n",
                    "print(\"Deep learning training completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 6: Ensemble Models"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Tao ensemble predictions\n",
                    "!python src/ensemble.py --config configs/config_baseline.yaml\n",
                    "\n",
                    "print(\"Ensemble creation completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 7: So sanh va phan tich"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# So sanh baseline vs DL models\n",
                    "!python compare_baselines.py\n",
                    "\n",
                    "print(\"Comparison analysis completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 8: Tao bao cao cho luan van"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Tao research summary\n",
                    "import sys\n",
                    "sys.path.append('src')\n",
                    "from research_summary import generate_research_report\n",
                    "generate_research_report()\n",
                    "\n",
                    "print(\"Research reports generated!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 9: Kiem tra ket qua"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Hien thi tom tat ket qua\n",
                    "print(\"RESEARCH COMPLETION SUMMARY\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "# Doc va hien thi summary\n",
                    "with open('reports/research_summary.txt', 'r', encoding='utf-8') as f:\n",
                    "    content = f.read()\n",
                    "    print(content[:1000])  # Hien thi 1000 ky tu dau\n",
                    "    print(\"...\")\n",
                    "    print(f\"\\nFull summary available at: reports/research_summary.txt\")\n",
                    "    print(f\"LaTeX tables available at: reports/latex_tables/\")\n",
                    "    print(f\"Detailed results available at: reports/\")\n",
                    "\n",
                    "print(\"Ready for thesis writing!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Buoc 10: Download ket qua"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Tao zip file de download\n",
                    "import shutil\n",
                    "\n",
                    "# Zip toan bo reports folder\n",
                    "shutil.make_archive('thesis_results', 'zip', 'reports')\n",
                    "\n",
                    "print(\"Results zipped as: thesis_results.zip\")\n",
                    "print(\"Download file nay ve may de su dung cho luan van\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save notebook
    notebook_path = Path("colab_research_pipeline.ipynb")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)

    print(f"Google Colab notebook created: {notebook_path}")
    print("Copy notebook nay len Google Colab va chay tung cell")

def create_colab_readme():
    """Create README for Google Colab usage"""

    readme_content = """# CHUNGKHOAN AI - Google Colab Guide

## **Huong Dan Chay Pipeline Tren Google Colab**

---

## **Buoc 1: Chuan Bi**

### **1.1 Upload Files Len Colab**
```python
# Upload cac file can thiet:
# - src/ (thu muc chua code)
# - configs/ (thu muc config)
# - requirements.txt
```

### **1.2 Mount Google Drive** (khuyen dung)
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy files tu Drive vao Colab
!cp -r /content/drive/MyDrive/ChungKhoanAI/* /content/
```

---

## **Buoc 2: Chay Notebook**

1. **Mo notebook**: `colab_research_pipeline.ipynb`
2. **Runtime**: Chon GPU neu co (Runtime → Change runtime type → GPU)
3. **Chay tung cell** theo thu tu tu tren xuong duoi

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

## **Tuy Chinh Cho Colab**

### **Neu gap loi Memory:**
- Giam batch_size trong config
- Su dung model nho hon
- Chay tung ticker mot

### **Neu Colab timeout:**
- Luu progress thuong xuyen
- Chia pipeline thanh nhieu notebook nho

---

## **Pipeline Steps:**

1. **Data Collection** - Thu thap tu vnstock API
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
"""

    readme_path = Path("COLAB_README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Colab guide created: {readme_path}")

def main():
    """Main function"""
    create_colab_notebook()
    create_colab_readme()

    print("\nGoogle Colab setup completed!")
    print("Files created:")
    print("  - colab_research_pipeline.ipynb")
    print("  - COLAB_README.md")
    print("\nNext steps:")
    print("  1. Copy files to Google Colab")
    print("  2. Follow COLAB_README.md instructions")
    print("  3. Run the notebook step by step")

if __name__ == "__main__":
    main()
