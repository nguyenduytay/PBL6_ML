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

## 🎯 Sử Dụng Model Sau Khi Train

Sau khi training thành công, model được lưu tại `ember_model_pycharm.txt` (hoặc `ember_model_2018.txt` nếu dùng script cũ).

### Cách 1: Load Model và Dự Đoán Đơn Giản

```python
import ember
import lightgbm as lgb

# Load model đã train
model = lgb.Booster(model_file="ember_model_pycharm.txt")

# Đọc file PE và dự đoán
file_path = r"C:\path\to\file.exe"
with open(file_path, "rb") as f:
    file_data = f.read()

# Dự đoán (score từ 0-1, >0.5 = malware)
score = ember.predict_sample(model, file_data, feature_version=2)
print(f"Malware probability: {score:.4f}")
print(f"Prediction: {'Malware' if score > 0.5 else 'Benign'}")
```

### Cách 2: Sử Dụng Script Có Sẵn

EMBER có sẵn script `classify_binaries.py` để phân tích file:

```bash
python scripts/classify_binaries.py -m ember_model_pycharm.txt -v 2 file1.exe file2.exe
```

### Cách 3: Phân Tích Nhiều File Trong Thư Mục

```python
import os
import ember
import lightgbm as lgb

# Load model
model = lgb.Booster(model_file="ember_model_pycharm.txt")

# Phân tích thư mục
directory = r"C:\samples"
results = []

for filename in os.listdir(directory):
    if filename.endswith(('.exe', '.dll', '.sys')):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
            score = ember.predict_sample(model, file_data, feature_version=2)

            results.append({
                'file': filename,
                'score': score,
                'prediction': 'Malware' if score > 0.5 else 'Benign'
            })
            print(f"{filename}: {results[-1]['prediction']} ({score:.4f})")
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")

# Lưu kết quả
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("malware_predictions.csv", index=False)
```

### Cách 4: Tích Hợp Vào API (Flask/FastAPI)

**Ví dụ với Flask:**

```python
from flask import Flask, request, jsonify
import ember
import lightgbm as lgb

app = Flask(__name__)
model = lgb.Booster(model_file="ember_model_pycharm.txt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_data = file.read()

    try:
        score = ember.predict_sample(model, file_data, feature_version=2)
        return jsonify({
            'malware_probability': float(score),
            'prediction': 'Malware' if score > 0.5 else 'Benign'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### ⚙️ Tham Số Quan Trọng

- **`feature_version=2`**: Luôn dùng version 2 (mặc định của EMBER2018)
- **Score**: Giá trị từ 0.0 đến 1.0
  - `score > 0.5`: Malware
  - `score <= 0.5`: Benign
- **File input**: Phải là file PE hợp lệ (.exe, .dll, .sys, v.v.)

### 📊 Kết Quả Training Mẫu

Với dataset EMBER2018, model đạt:

- **Accuracy**: ~94%
- **Precision**: ~98%
- **Recall**: ~90%
- **F1-Score**: ~94%
- **AUC**: ~0.99

### 🔍 Kiểm Tra Model

```python
import lightgbm as lgb

# Load model
model = lgb.Booster(model_file="ember_model_pycharm.txt")

# Xem thông tin model
print(f"Number of trees: {model.num_trees()}")
print(f"Number of features: {model.num_feature()}")
```
