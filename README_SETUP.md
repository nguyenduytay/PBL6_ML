# 🛡️ EMBER MALWARE DETECTION - SETUP GUIDE

## 📁 Cấu trúc file sau khi setup

```
D:\pbl6\ember\
├── 📄 HUONG_DAN_CHAY_DU_AN.md      # Hướng dẫn chi tiết
├── 📄 QUICK_START.md               # Hướng dẫn nhanh
├── 📄 README_SETUP.md              # File này
├── 🐳 Dockerfile                   # Docker configuration
├── 🐳 docker-compose.yml          # Docker Compose config
├── 🐍 ember_demo.py               # Demo script
├── 🐍 test_ember.py               # Test script
├── 🐍 run_ember_docker.bat        # Script chạy Docker
├── 🐍 setup_ember.bat             # Script setup tự động
├── 📊 ember/                      # EMBER source code
├── 📊 scripts/                    # Utility scripts
├── 📊 malconv/                    # MalConv model
└── 📊 resources/                  # Notebooks và resources
```

## 🚀 CÁCH CHẠY NHANH

### Phương án 1: Tự động setup

```bash
# Chạy script setup tự động
setup_ember.bat
```

### Phương án 2: Manual setup

```bash
# 1. Build Docker image
docker build -t ember-malware-detection .

# 2. Chạy container
docker run -it --rm -v "%cd%":/workspace ember-malware-detection /bin/bash

# 3. Test EMBER
python -c "import ember; print('EMBER OK!')"
```

### Phương án 3: Docker Compose

```bash
# Chạy với Docker Compose
docker-compose up -d

# Vào container
docker exec -it ember-malware-detection /bin/bash
```

## 🔧 CÁC SCRIPT CÓ SẴN

### 1. Test Scripts

- `test_ember.py` - Kiểm tra cài đặt EMBER
- `ember_demo.py` - Demo phân tích file PE

### 2. Setup Scripts

- `setup_ember.bat` - Tự động setup môi trường
- `run_ember_docker.bat` - Chạy Docker container

### 3. Utility Scripts

- `scripts/classify_binaries.py` - Phân tích file với model
- `scripts/init_ember.py` - Huấn luyện model

## 📊 CÁCH SỬ DỤNG

### 1. Phân tích file PE

```python
import ember

# Tạo feature extractor
extractor = ember.PEFeatureExtractor(2)

# Đọc file PE
with open("sample.exe", "rb") as f:
    file_data = f.read()

# Trích xuất features
features = extractor.feature_vector(file_data)
print(f"Extracted {len(features)} features")
```

### 2. Sử dụng pre-trained model

```python
import ember
import lightgbm as lgb

# Load model
model = lgb.Booster(model_file="ember_model.txt")

# Phân tích file
prediction = ember.predict_sample(model, file_data)
print(f"Malware probability: {prediction}")
```

### 3. Huấn luyện model mới

```python
import ember

# Tạo vectorized features
ember.create_vectorized_features("/data/ember/")

# Huấn luyện model
model = ember.train_model("/data/ember/")

# Lưu model
model.save_model("my_model.txt")
```

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

## 📈 HIỆU SUẤT

### Model Performance:

- **ROC AUC**: > 0.99
- **False Positive Rate**: < 1%
- **Detection Rate**: > 95%
- **Speed**: Vài giây/file

### Features:

- **Total**: 2381 features
- **Byte-level**: 416 features
- **PE Structure**: 1861 features
- **Strings**: 104 features

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

## 📚 TÀI LIỆU THAM KHẢO

- **EMBER Paper**: https://arxiv.org/abs/1804.04637
- **GitHub**: https://github.com/elastic/ember
- **LIEF Library**: https://github.com/lief-project/LIEF
- **LightGBM**: https://lightgbm.readthedocs.io/

## 🆘 HỖ TRỢ

Nếu gặp vấn đề:

1. Chạy `test_ember.py` để kiểm tra
2. Xem `HUONG_DAN_CHAY_DU_AN.md` để biết chi tiết
3. Kiểm tra Docker logs: `docker logs ember-malware-detection`
4. Tham khảo GitHub issues: https://github.com/elastic/ember/issues

---

**Chúc bạn sử dụng EMBER thành công! 🎉**
