# Hướng dẫn Setup và Chạy EMBER Project

## 📋 Requirements

- Python 3.8+ (đang dùng Python 3.12.10)
- Các thư viện trong `requirements.txt`

## 🔧 Cài đặt

### Option 1: Dùng Virtual Environment (Khuyến nghị)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt trên Windows
venv\Scripts\activate

# Kích hoạt trên Linux/Mac
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Option 2: Cài trực tiếp (Không khuyến nghị)

```bash
pip install -r requirements.txt
```

**Lưu ý**: Có thể gây conflict với các packages khác trong hệ thống.

## 🚀 Chạy Code

### Cách 1: Dùng script helper

```bash
python scripts/init_ember.py data/ember2018 --train
```

### Cách 2: Chạy trực tiếp với PYTHONPATH

**Windows:**

```powershell
$env:PYTHONPATH="$PWD"; python scripts/init_ember.py data/ember2018 --train
```

**Linux/Mac:**

```bash
PYTHONPATH=. python scripts/init_ember.py data/ember2018 --train
```

### Cách 3: Chạy từ thư mục root

```bash
cd d:\pbl6\ember
python -m scripts.init_ember data/ember2018 --train
```

## 📊 Dataset Hiện Có

Dataset trong `data/ember2018/` đã có sẵn:

- ✅ `X_train.dat` (7.1GB) - Vectorized features
- ✅ `y_train.dat` (3.1MB) - Labels
- ✅ `ember_model_2018.txt` (121MB) - Model đã train
- ✅ `train_features_*.jsonl` - Raw features

Vì đã có `X_train.dat` và `y_train.dat`, code sẽ **bỏ qua** bước tạo vectorized features và có thể train ngay.

## 📝 Các Lệnh Thường Dùng

### Chỉ train model mới:

```bash
python scripts/init_ember.py data/ember2018 --train
```

### Tạo metadata:

```bash
python scripts/init_ember.py data/ember2018 --metadata
```

### Optimize parameters trước khi train:

```bash
python scripts/init_ember.py data/ember2018 --train --optimize
```

## ⚠️ Lưu Ý Quan Trọng

1. **File size lớn**: Dataset không được commit vào git (đã có trong `.gitignore`)
2. **Memory**: Cần ít nhất 8GB RAM để load dataset
3. **Dependencies**: Phải cài đúng version `lief==0.9.0` để tương thích với EMBER v2

## 🐛 Troubleshooting

### Lỗi `ModuleNotFoundError: No module named 'ember'`

**Giải pháp**: Set PYTHONPATH hoặc dùng `init_ember.py`

```bash
python scripts/init_ember.py data/ember2018 --train
```

### Lỗi import dependencies

```bash
pip install -r requirements.txt
```

### LIEF version warning

Kiểm tra version:

```bash
python -c "import lief; print(lief.__version__)"
```

Phải là `0.9.0` để tương thích với EMBER feature version 2.
