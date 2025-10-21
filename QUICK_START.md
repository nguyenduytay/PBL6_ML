# ⚡ QUICK START - EMBER MALWARE DETECTION

## 🚀 Chạy nhanh trong 3 bước

### Bước 1: Khởi động Docker

```bash
# Đảm bảo Docker Desktop đang chạy
docker ps
```

### Bước 2: Build và chạy

```bash
# Build image
docker build -t ember-malware-detection .

# Chạy container
docker run -it --rm -v "%cd%":/workspace ember-malware-detection /bin/bash
```

### Bước 3: Phân tích file

```bash
# Trong container, chạy:
python /workspace/ember_demo.py /workspace/your_file.exe
```

---

## 📋 Các lệnh hữu ích

### Kiểm tra cài đặt:

```bash
python -c "import ember; print('EMBER OK!')"
```

### Phân tích file đơn:

```bash
python scripts/classify_binaries.py -m model.txt file.exe
```

### Tạo features:

```python
import ember
extractor = ember.PEFeatureExtractor(2)
features = extractor.feature_vector(file_data)
```

### Huấn luyện model:

```python
import ember
ember.create_vectorized_features("/data/ember/")
model = ember.train_model("/data/ember/")
```

---

## 🔧 Scripts có sẵn

- `run_ember_docker.bat` - Chạy Docker container
- `ember_demo.py` - Demo phân tích file PE
- `scripts/classify_binaries.py` - Phân tích file với model
- `scripts/init_ember.py` - Huấn luyện model

---

## 📊 Kết quả mong đợi

- **Malware probability**: 0.0 (benign) đến 1.0 (malicious)
- **Features extracted**: 2381 features
- **Processing time**: Vài giây/file
- **Accuracy**: > 95%

---

**Xem file `HUONG_DAN_CHAY_DU_AN.md` để biết thêm chi tiết! 📚**
