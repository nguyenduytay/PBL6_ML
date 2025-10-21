# 🛡️ HƯỚNG DẪN CHẠY DỰ ÁN EMBER - MALWARE DETECTION

## 📋 Tổng quan dự án

**EMBER (Elastic Malware Benchmark for Empowering Researchers)** là một framework hoàn chỉnh để phát hiện malware sử dụng Machine Learning, đặc biệt tập trung vào file PE (Portable Executable) của Windows.

### 🎯 Mục đích chính:

- **Phát hiện malware tự động** sử dụng AI/ML
- **Tạo benchmark dataset** cho cộng đồng nghiên cứu
- **So sánh hiệu quả** giữa các phương pháp ML khác nhau
- **Tích hợp vào hệ thống bảo mật** thực tế

---

## 🚀 CÁCH 1: CHẠY BẰNG DOCKER (KHUYẾN NGHỊ)

### Bước 1: Cài đặt Docker Desktop

1. Download Docker Desktop từ: https://www.docker.com/products/docker-desktop/
2. Cài đặt và khởi động Docker Desktop
3. Đảm bảo Docker đang chạy (icon Docker trong system tray)

### Bước 2: Build Docker Image

```bash
# Di chuyển vào thư mục dự án
cd D:\pbl6\ember

# Build Docker image
docker build -t ember-malware-detection .
```

### Bước 3: Chạy container

```bash
# Chạy container với shell tương tác
docker run -it --rm -v "%cd%":/workspace ember-malware-detection /bin/bash

# Hoặc sử dụng script đã tạo
run_ember_docker.bat
```

### Bước 4: Sử dụng EMBER trong container

```bash
# Kiểm tra cài đặt
python -c "import ember; print('EMBER OK!')"

# Phân tích file PE
python ember_demo.py /workspace/your_file.exe
```

---

## 🐍 CÁCH 2: CHẠY TRỰC TIẾP TRÊN WINDOWS

### Bước 1: Cài đặt Python dependencies

```bash
# Cài đặt từ requirements
pip install -r requirements.txt

# Hoặc cài đặt EMBER trực tiếp
pip install git+https://github.com/elastic/ember.git
```

### Bước 2: Cài đặt EMBER

```bash
# Cài đặt EMBER package
python setup.py install
```

### Bước 3: Kiểm tra cài đặt

```python
import ember
print("EMBER installed successfully!")
```

---

## 📊 CÁCH SỬ DỤNG EMBER

### 1. Phân tích file PE đơn lẻ

```python
import ember
import lightgbm as lgb

# Load model đã train
model = lgb.Booster(model_file="path/to/model.txt")

# Phân tích file PE
with open("suspicious.exe", "rb") as f:
    file_data = f.read()

# Dự đoán
prediction = ember.predict_sample(model, file_data)
print(f"Malware probability: {prediction}")
```

### 2. Trích xuất features từ file PE

```python
import ember

# Tạo feature extractor
extractor = ember.PEFeatureExtractor(feature_version=2)

# Đọc file PE
with open("sample.exe", "rb") as f:
    file_data = f.read()

# Trích xuất features
features = extractor.feature_vector(file_data)
print(f"Extracted {len(features)} features")
```

### 3. Huấn luyện model mới

```python
import ember

# Tạo vectorized features từ raw data
ember.create_vectorized_features("/path/to/dataset/")

# Huấn luyện model
model = ember.train_model("/path/to/dataset/")

# Lưu model
model.save_model("my_ember_model.txt")
```

---

## 📁 CẤU TRÚC DỮ LIỆU

### Dataset EMBER bao gồm:

- **EMBER 2017**: 1.1M file PE (900K train + 200K test)
- **EMBER 2018**: 1M file PE (800K train + 200K test)

### Cấu trúc thư mục dataset:

```
/data/ember/
├── train_features_0.jsonl    # Raw features (training)
├── train_features_1.jsonl
├── ...
├── test_features.jsonl       # Raw features (testing)
├── X_train.dat              # Vectorized features (training)
├── y_train.dat              # Labels (training)
├── X_test.dat               # Vectorized features (testing)
├── y_test.dat               # Labels (testing)
├── metadata.csv             # Metadata
└── ember_model_2018.txt    # Pre-trained model
```

---

## 🔧 CÁC SCRIPT CÓ SẴN

### 1. classify_binaries.py

```bash
# Phân tích file PE với model đã train
python scripts/classify_binaries.py -m model.txt file1.exe file2.exe

# Sử dụng feature version 2
python scripts/classify_binaries.py -v 2 -m model.txt file.exe
```

### 2. init_ember.py

```bash
# Tạo vectorized features
python scripts/init_ember.py /path/to/dataset/

# Tạo metadata
python scripts/init_ember.py -m /path/to/dataset/

# Huấn luyện model
python scripts/init_ember.py -t /path/to/dataset/

# Tối ưu hóa parameters
python scripts/init_ember.py -t --optimize /path/to/dataset/
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

## 🆘 TROUBLESHOOTING

### Lỗi thường gặp:

#### 1. LIEF installation error:

```bash
# Sử dụng Docker thay vì cài đặt trực tiếp
docker run -it ember-malware-detection /bin/bash
```

#### 2. Memory error:

```bash
# Giảm batch size hoặc sử dụng máy có RAM lớn hơn
# Hoặc sử dụng Docker với memory limit
docker run -m 8g ember-malware-detection
```

#### 3. File not found:

```bash
# Kiểm tra đường dẫn file
# Đảm bảo file PE tồn tại và có quyền đọc
```

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề, hãy:

1. Kiểm tra log lỗi chi tiết
2. Đảm bảo Docker đang chạy
3. Kiểm tra quyền truy cập file
4. Tham khảo GitHub issues: https://github.com/elastic/ember/issues

---

**Chúc bạn sử dụng EMBER thành công! 🎉**
