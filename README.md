# 🛡️ EMBER MALWARE DETECTION - COMPLETE GUIDE

## 📋 Tổng quan dự án

**EMBER (Elastic Malware Benchmark for Empowering Researchers)** là một hệ thống phát hiện malware tự động sử dụng Machine Learning, được phát triển bởi Elastic Security.

### 🎯 Dự án này giải quyết vấn đề gì:

- **Phát hiện malware tự động**: Tự động phân loại file PE (Windows executable) là malware hay benign
- **Bảo mật hệ thống**: Bảo vệ máy tính khỏi các phần mềm độc hại
- **Nghiên cứu AI/ML**: Cung cấp benchmark dataset cho cộng đồng nghiên cứu
- **Ứng dụng thực tế**: Tích hợp vào hệ thống antivirus, email security, endpoint protection

### 🔍 Cách hoạt động:

1. **Input**: File PE (.exe, .dll, .sys) của Windows
2. **Feature Extraction**: Trích xuất 2381 features từ PE file (headers, sections, imports, strings...)
3. **Machine Learning**: Sử dụng LightGBM để phân loại
4. **Output**: Xác suất malware (0.0 = benign, 1.0 = malicious)

---

## ⚡ QUICK START - CÁCH CHẠY NHANH

### 🚀 Phương án 1: Chạy trực tiếp (Khuyến nghị)

```bash
# 1. Chạy training script
python colab_guide/ember_pycharm.py

# 2. Đợi training hoàn tất (30-60 phút)
# 3. Model sẽ được lưu: colab_guide/ember_model_pycharm.txt
```

### 🐳 Phương án 2: Sử dụng Docker

```bash
# 1. Khởi động Docker Desktop
# 2. Build image
docker build -t ember-malware-detection .

# 3. Chạy container
docker run -it --rm -v "%cd%":/workspace ember-malware-detection /bin/bash

# 4. Trong container, chạy training
python /workspace/colab_guide/ember_pycharm.py
```

### 📊 Phương án 3: Google Colab

1. Mở file `colab_guide/ember_colab_notebook.ipynb`
2. Upload lên Google Colab
3. Chạy tất cả cells
4. Training sẽ chạy trên GPU miễn phí

---

## 🚀 HƯỚNG DẪN CHI TIẾT

### 📋 Yêu cầu hệ thống

- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **Storage**: 50GB trống
- **Python**: 3.8+ (khuyến nghị 3.10)
- **OS**: Windows 10/11, Linux, macOS

### 🎯 Các phương án chạy

#### 1. 🚀 Chạy trực tiếp (Đơn giản nhất)

```bash
# Bước 1: Đảm bảo có Python 3.8+
python --version

# Bước 2: Chạy training script
python colab_guide/ember_pycharm.py

# Bước 3: Đợi training hoàn tất
# - Loading data: 5-10 phút
# - Training: 30-60 phút
# - Tổng cộng: 45-70 phút
```

#### 2. 🐳 Sử dụng Docker (Ổn định nhất)

```bash
# Bước 1: Cài đặt Docker Desktop
# Download từ: https://www.docker.com/products/docker-desktop/

# Bước 2: Build image
docker build -t ember-malware-detection .

# Bước 3: Chạy container
docker run -it --rm -v "%cd%":/workspace ember-malware-detection /bin/bash

# Bước 4: Trong container
python /workspace/colab_guide/ember_pycharm.py
```

#### 3. 📊 Google Colab (Miễn phí GPU)

1. Mở [Google Colab](https://colab.research.google.com/)
2. Upload file `colab_guide/ember_colab_notebook.ipynb`
3. Chạy tất cả cells
4. Training sẽ chạy trên GPU T4 miễn phí

---

## 📊 CÁCH SỬ DỤNG MODEL SAU KHI TRAINING

### 🎯 Sử dụng model đã train

```python
import lightgbm as lgb
import ember

# 1. Load model đã train
model = lgb.Booster(model_file="train/ember_model_pycharm.txt")

# 2. Phân tích file PE
def analyze_file(file_path):
    score = ember.predict_sample(model, file_path, feature_version=2)
    return score

# 3. Test với file
score = analyze_file("test_file.exe")
print(f"Malware probability: {score:.4f}")
print(f"Prediction: {'Malware' if score > 0.5 else 'Benign'}")
```

### 🔍 Batch analysis nhiều file

```python
import os
import lightgbm as lgb
import ember

# Load model
model = lgb.Booster(model_file="train/ember_model_pycharm.txt")

# Phân tích thư mục
def analyze_directory(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(('.exe', '.dll', '.sys')):
            file_path = os.path.join(directory, filename)
            try:
                score = ember.predict_sample(model, file_path, feature_version=2)
                results.append({
                    'file': filename,
                    'malware_prob': score,
                    'prediction': 'Malware' if score > 0.5 else 'Benign'
                })
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
    return results

# Sử dụng
results = analyze_directory("path/to/pe/files/")
for result in results:
    print(f"{result['file']}: {result['prediction']} ({result['malware_prob']:.4f})")
```

### 📈 Hiệu suất model

- **Accuracy**: > 95%
- **ROC AUC**: > 0.99
- **False Positive Rate**: < 1%
- **Speed**: Vài giây/file

---

## 📁 CẤU TRÚC DỰ ÁN

### 📂 Thư mục chính:

```
ember/
├── colab_guide/                    # Training scripts
│   ├── ember_pycharm.py           # Script chạy trên PyCharm/VSCode
│   ├── ember_colab_notebook.ipynb # Script chạy trên Google Colab
│   ├── ember_model_pycharm.txt    # Model đã train (sẽ tạo)
│   └── ember_training.log         # Log file training
├── data/ember2018/                # Dataset EMBER2018
│   ├── train_features_0.jsonl    # Features training
│   ├── train_features_1.jsonl
│   ├── ...
│   ├── train_features_5.jsonl
│   └── test_features.jsonl        # Features testing
├── ember/                         # EMBER source code
│   ├── __init__.py
│   └── features.py
├── scripts/                       # Utility scripts
│   ├── classify_binaries.py
│   └── init_ember.py
└── README.md                      # Hướng dẫn này
```

### 📊 Dataset EMBER2018:

- **Kích thước**: 1M file PE
- **Features**: 2381 features/file
- **Labels**: 0 (benign) hoặc 1 (malware)
- **Format**: JSONL files

---

## 🔧 CÁC SCRIPT TRAINING

### 1. 🚀 ember_pycharm.py (Khuyến nghị)

```bash
# Chạy training trên PyCharm/VSCode
python colab_guide/ember_pycharm.py

# Script sẽ:
# - Tự động cài đặt dependencies
# - Load dataset từ data/ember2018/
# - Training LightGBM model
# - Lưu model: colab_guide/ember_model_pycharm.txt
```

### 2. 📊 ember_colab_notebook.ipynb

```bash
# Upload lên Google Colab
# Chạy tất cả cells
# Training trên GPU miễn phí
```

### 3. 🛠️ Utility scripts

```bash
# Phân tích file PE với model đã train
python scripts/classify_binaries.py -m colab_guide/ember_model_pycharm.txt file.exe

# Tạo metadata (nếu cần)
python scripts/init_ember.py -m data/ember2018/
```

---

## 📈 HIỆU SUẤT MODEL

### EMBER Model Performance:

- **ROC AUC**: > 0.99
- **False Positive Rate**: < 1%
- **Detection Rate**: > 95%
- **Speed**: Vài giây/file

### So sánh với MalConv:

- **EMBER (LightGBM)**: Nhanh, chính xác cao
- **MalConv (CNN)**: Chậm hơn, cần GPU

---

## 🔍 FEATURES ĐƯỢC TRÍCH XUẤT

### 1. Byte-level Features (416 features):

- **ByteHistogram**: Phân bố byte (256 features)
- **ByteEntropyHistogram**: Entropy của byte (256 features)

### 2. String Features (104 features):

- **StringExtractor**: Chuỗi trong file
- **Paths, URLs, Registry**: Các pattern đặc biệt

### 3. PE Structure Features (1861 features):

- **GeneralFileInfo**: Thông tin chung (10 features)
- **HeaderFileInfo**: Thông tin header (62 features)
- **SectionInfo**: Thông tin sections (255 features)
- **ImportsInfo**: Thư viện import (1280 features)
- **ExportsInfo**: Hàm export (128 features)
- **DataDirectories**: Data directories (30 features)

**Tổng cộng: 2381 features**

---

## 🛠️ TÍCH HỢP VÀO HỆ THỐNG

### 1. API Service

```python
from flask import Flask, request, jsonify
import ember
import lightgbm as lgb

app = Flask(__name__)
model = lgb.Booster(model_file="ember_model.txt")

@app.route('/analyze', methods=['POST'])
def analyze_file():
    file_data = request.files['file'].read()
    prediction = ember.predict_sample(model, file_data)
    return jsonify({'malware_probability': prediction})

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. Batch Processing

```python
import os
import ember

def analyze_directory(directory_path, model):
    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.exe'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as f:
                file_data = f.read()
            prediction = ember.predict_sample(model, file_data)
            results.append({'file': filename, 'malware_prob': prediction})
    return results
```

---

## 🐳 DOCKER COMMANDS

### Build và chạy

```bash
# Build image
docker build -t ember-malware-detection .

# Chạy container
docker run -it --rm -v "%cd%":/workspace ember-malware-detection /bin/bash

# Chạy với Docker Compose
docker-compose up -d
```

### Quản lý container

```bash
# Xem containers đang chạy
docker ps

# Vào container
docker exec -it ember-malware-detection /bin/bash

# Dừng container
docker-compose down
```

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers:

1. **EMBER Paper**: https://arxiv.org/abs/1804.04637
2. **MalConv Paper**: https://arxiv.org/abs/1710.09435

### Datasets:

- **EMBER 2017**: https://ember.elastic.co/ember_dataset.tar.bz2
- **EMBER 2018**: https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

### GitHub:

- **EMBER Repository**: https://github.com/elastic/ember
- **LIEF Library**: https://github.com/lief-project/LIEF

---

## 🚨 LƯU Ý QUAN TRỌNG

### 1. Bảo mật:

- **Chỉ phân tích file PE**: EMBER chỉ hoạt động với file PE
- **Sandbox environment**: Chạy trong Docker để an toàn
- **Quét virus trước**: Kiểm tra file trước khi phân tích

### 2. Hiệu suất:

- **Memory usage**: Cần ít nhất 8GB RAM cho dataset lớn
- **CPU intensive**: Quá trình training cần CPU mạnh
- **Storage**: Dataset cần ~50GB dung lượng

### 3. Tương thích:

- **Python 3.8+**: Khuyến nghị sử dụng Python 3.8
- **LIEF 0.9.0**: Phiên bản cố định để đảm bảo tính nhất quán
- **Windows/Linux**: Hoạt động trên cả hai hệ điều hành

---

## 🔍 TROUBLESHOOTING

### Lỗi thường gặp:

#### 1. Docker không chạy

```bash
# Khởi động Docker Desktop
# Kiểm tra: docker ps
```

#### 2. LIEF installation error

```bash
# Sử dụng Docker thay vì cài đặt trực tiếp
docker run -it ember-malware-detection /bin/bash
```

#### 3. Memory error

```bash
# Tăng memory limit cho Docker
docker run -m 8g ember-malware-detection
```

#### 4. File not found

```bash
# Kiểm tra đường dẫn file
# Đảm bảo file PE tồn tại và có quyền đọc
```

---

## 🆘 HỖ TRỢ

Nếu gặp vấn đề:

1. Kiểm tra log lỗi chi tiết
2. Đảm bảo Docker đang chạy
3. Kiểm tra quyền truy cập file
4. Tham khảo GitHub issues: https://github.com/elastic/ember/issues

---

## 📊 KẾT QUẢ MONG ĐỢI

- **Malware probability**: 0.0 (benign) đến 1.0 (malicious)
- **Features extracted**: 2381 features
- **Processing time**: Vài giây/file
- **Accuracy**: > 95%

---

**Chúc bạn sử dụng EMBER thành công! 🎉**
