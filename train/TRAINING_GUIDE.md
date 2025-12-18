# 🚀 HƯỚNG DẪN TRAIN EMBER MODEL - V1 & V2

Hướng dẫn chi tiết cách train 2 model EMBER: **ember_v1.py** (chuẩn) và **ember_v2.py** (linh hoạt).

---

## 📋 MỤC LỤC

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Chuẩn bị môi trường](#2-chuẩn-bị-môi-trường)
3. [Train Model V1 (Chuẩn)](#3-train-model-v1-ember_v1py-chuẩn)
4. [Train Model V2 (Linh hoạt)](#4-train-model-v2-ember_v2py-linh-hoạt)
5. [So sánh V1 vs V2](#5-so-sánh-v1-vs-v2)
6. [Test Model sau khi train](#6-test-model-sau-khi-train)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. YÊU CẦU HỆ THỐNG

### Phần cứng:
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **Storage**: 50GB trống (cho dataset EMBER2018)
- **CPU**: Multi-core (4+ cores khuyến nghị)

### Phần mềm:
- **Python**: 3.8+ (khuyến nghị Python 3.10)
- **OS**: Windows 10/11, Linux, macOS
- **PowerShell** (Windows) hoặc **Terminal** (Linux/Mac)

### Dataset:
Dataset EMBER2018 phải có trong thư mục `data/ember2018/`:

**File bắt buộc:**
- `train_features_0.jsonl` đến `train_features_5.jsonl` (6 files)
- `test_features.jsonl`

**File tùy chọn (nếu đã vectorize):**
- `X_train.dat`, `y_train.dat`
- `X_test.dat`, `y_test.dat`

---

## 2. CHUẨN BỊ MÔI TRƯỜNG

### Bước 1: Tạo virtual environment

**Windows PowerShell:**
```powershell
cd D:\pbl6\ember
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
```

**Linux/Mac:**
```bash
cd /path/to/ember
python3.10 -m venv venv
source venv/bin/activate
python -m pip install -U pip
```

### Bước 2: Cài đặt dependencies

```powershell
pip install tqdm numpy pandas lightgbm scikit-learn lief==0.12.3 psutil
```

**Lưu ý quan trọng:**
- Dùng `lief==0.12.3` để tương thích với Python 3.10+
- Phiên bản `lief==0.9.0` không có wheel cho Python 3.10/3.12

### Bước 3: Kiểm tra dataset

```powershell
# Kiểm tra dataset có đầy đủ không
dir data\ember2018\train_features_*.jsonl
dir data\ember2018\test_features.jsonl
```

**Kết quả mong đợi:**
```
train_features_0.jsonl
train_features_1.jsonl
train_features_2.jsonl
train_features_3.jsonl
train_features_4.jsonl
train_features_5.jsonl
test_features.jsonl
```

---

## 3. TRAIN MODEL V1 (`ember_v1.py`) - CHUẨN

### 🎯 Khi nào dùng V1?

- ✅ Dataset **chuẩn EMBER2018** (đầy đủ fields, đúng format)
- ✅ Cần model **chính xác, tin cậy** (cho báo cáo, production)
- ✅ Muốn **tối ưu RAM** (dùng memory-mapped files)
- ✅ Muốn **bám sát paper EMBER** (đúng pipeline)

### 📝 Cách train V1:

**Bước 1: Chạy script train**

```powershell
cd D:\pbl6\ember
.\venv\Scripts\Activate.ps1
python -m train.ember_v1
```

**Hoặc chạy trực tiếp:**
```powershell
python train\ember_v1.py
```

**Bước 2: Quá trình training**

Script sẽ tự động:

1. ✅ **Kiểm tra yêu cầu hệ thống** (Python version, RAM)
2. ✅ **Cài đặt dependencies** (nếu thiếu)
3. ✅ **Setup EMBER repository** (clone nếu chưa có)
4. ✅ **Kiểm tra dataset**:
   - Có `train_features_*.jsonl` và `test_features.jsonl`
   - Kiểm tra format: có `label`, đủ fields (`histogram`, `byteentropy`, `general`, `header`, `section`, `imports`, `exports`, `strings`)
5. ✅ **Tạo vectorized features** (nếu chưa có):
   - Gọi `ember.create_vectorized_features(...)`
   - Tạo `X_train.dat`, `y_train.dat`, `X_test.dat`, `y_test.dat`
6. ✅ **Load vectorized features** (memory-mapped, tiết kiệm RAM)
7. ✅ **Train LightGBM model**:
   - Parameters tối ưu cho dataset lớn
   - Early stopping để tránh overfitting
   - Validation trên test set
8. ✅ **Đánh giá model** (accuracy, precision, recall, F1, AUC)
9. ✅ **Lưu model**: `ember_model_pycharm.txt`

**Bước 3: Kết quả**

Sau khi train xong, bạn sẽ có:

- ✅ **Model**: `D:\pbl6\ember\ember_model_pycharm.txt`
- ✅ **Log**: `D:\pbl6\ember\ember_training.log`

**Thời gian training:**
- Vectorize features: 10-30 phút (nếu chưa có `.dat`)
- Training: 30-60 phút
- **Tổng cộng**: 45-90 phút

### ✅ Ưu điểm V1:

1. **Chuẩn nhất**: Bám sát paper EMBER, đúng pipeline
2. **Tối ưu RAM**: Dùng memory-mapped files (`.dat`), không load toàn bộ vào RAM
3. **Kiểm tra chặt chẽ**: Đảm bảo dataset đúng format → model tin cậy
4. **Kết quả chính xác**: Dùng label thật từ dataset
5. **Phù hợp production**: Model đủ tin cậy để deploy

### ❌ Nhược điểm V1:

1. **Yêu cầu dataset chuẩn**: Nếu thiếu field hoặc format sai → dừng lại
2. **Ít linh hoạt**: Không chấp nhận dataset "gần đúng"

---

## 4. TRAIN MODEL V2 (`ember_v2.py`) - LINH HOẠT

### 🎯 Khi nào dùng V2?

- ✅ Dataset **không chuẩn** (thiếu fields, format lệch)
- ✅ Chỉ cần **demo / test pipeline** (không cần model chính xác)
- ✅ Dataset **không có label** (sẽ tạo label giả)
- ✅ Muốn **train thử nhanh** mà không cần dataset hoàn hảo

### 📝 Cách train V2:

**Bước 1: Chạy script train**

```powershell
cd D:\pbl6\ember
.\venv\Scripts\Activate.ps1
python -m train.ember_v2
```

**Hoặc chạy trực tiếp:**
```powershell
python train\ember_v2.py
```

**Bước 2: Quá trình training**

Script sẽ tự động:

1. ✅ **Kiểm tra yêu cầu hệ thống** (Python version, RAM)
2. ✅ **Cài đặt dependencies** (nếu thiếu)
3. ✅ **Setup EMBER repository** (clone nếu chưa có)
4. ✅ **Kiểm tra dataset**:
   - Có `train_features_*.jsonl` (không bắt buộc đầy đủ fields)
   - **Không kiểm tra format chặt chẽ** như V1
5. ✅ **Tạo vectorized features** (nếu chưa có):
   - Thử gọi `ember.create_vectorized_features(...)`
   - **Nếu lỗi** → bỏ qua, đọc trực tiếp JSONL
6. ✅ **Load data từ JSONL** (nếu không có `.dat`):
   - Đọc toàn bộ `train_features_*.jsonl` vào RAM
   - Extract features (bỏ `sha256`, `md5`)
   - **Tạo label giả** nếu không có `label`:
     ```python
     # Label giả dựa trên hash
     label = 1 if sha256[-1] in '13579bdf' else 0
     ```
7. ✅ **Train/test split** (80/20)
8. ✅ **Train LightGBM model**
9. ✅ **Đánh giá model**
10. ✅ **Lưu model**: `ember_model_pycharm.txt`

**Bước 3: Kết quả**

Sau khi train xong, bạn sẽ có:

- ✅ **Model**: `D:\pbl6\ember\ember_model_pycharm.txt`
- ✅ **Log**: `D:\pbl6\ember\ember_training.log`

**Thời gian training:**
- Đọc JSONL: 5-15 phút (nếu không có `.dat`)
- Training: 30-60 phút
- **Tổng cộng**: 35-75 phút

### ✅ Ưu điểm V2:

1. **Linh hoạt**: Chấp nhận dataset không chuẩn, vẫn cố gắng train
2. **Dễ chạy**: Không cần dataset hoàn hảo
3. **Phù hợp demo**: Có thể train ngay cả khi thiếu label

### ❌ Nhược điểm V2:

1. **Model không chính xác**: Nếu dùng label giả → model không có ý nghĩa thực tế
2. **Tốn RAM**: Đọc toàn bộ JSONL vào RAM (không dùng memmap)
3. **Chậm hơn**: Đọc JSONL chậm hơn đọc `.dat`
4. **Không chuẩn**: Không bám sát paper EMBER

---

## 5. SO SÁNH V1 VS V2

### Bảng so sánh chi tiết:

| Tiêu chí | **V1 (ember_v1.py)** | **V2 (ember_v2.py)** |
|----------|---------------------|----------------------|
| **Dataset yêu cầu** | ✅ Chuẩn EMBER2018, đầy đủ fields | ⚠️ Dataset lệch chuẩn vẫn chạy được |
| **Kiểm tra format** | ✅ Rất chặt chẽ (đủ fields, đúng type) | ⚠️ Lỏng lẻo (chỉ cần có JSONL) |
| **Đọc dữ liệu** | ✅ Ưu tiên `.dat` (memory-mapped) | ⚠️ Đọc full JSONL vào RAM |
| **Label** | ✅ Dùng `label` thật từ dataset | ❌ Có thể tạo label giả |
| **Tốc độ** | ✅ Nhanh (memmap) | ⚠️ Chậm hơn (đọc JSONL) |
| **RAM usage** | ✅ Thấp (~8-16GB) | ⚠️ Cao (~16-32GB) |
| **Độ chính xác** | ✅ Cao (label thật) | ❌ Thấp (nếu dùng label giả) |
| **Độ tin cậy** | ✅ Rất cao (chuẩn paper) | ⚠️ Thấp (không chuẩn) |
| **Phù hợp** | ✅ Báo cáo, production | ⚠️ Demo, test pipeline |
| **Khi nào dùng** | ✅ Dataset chuẩn, cần model tốt | ⚠️ Dataset không chuẩn, chỉ demo |

### 🎯 Khuyến nghị:

- **Dùng V1** nếu:
  - Dataset của bạn **chuẩn EMBER2018**
  - Cần model **chính xác, tin cậy**
  - Làm **đồ án nghiêm túc** hoặc **production**

- **Dùng V2** nếu:
  - Dataset **không chuẩn** hoặc **thiếu fields**
  - Chỉ cần **demo / test pipeline**
  - Không quan tâm độ chính xác (chỉ muốn thấy train chạy)

---

## 6. TEST MODEL SAU KHI TRAIN

Sau khi train xong (dù V1 hay V2), bạn sẽ có model: `ember_model_pycharm.txt`

### Test 1 file:

```powershell
python -m train.test_ember_model -f C:\Windows\System32\notepad.exe
```

**Kết quả:**
```
2025-12-16 18:00:00 - INFO - ✓ Model đã load thành công
2025-12-16 18:00:00 - INFO -   - Số cây: 500
2025-12-16 18:00:00 - INFO -   - Số features: 2,381
2025-12-16 18:00:01 - INFO - Đang phân tích: notepad.exe
2025-12-16 18:00:02 - INFO - KẾT QUẢ PHÂN TÍCH
2025-12-16 18:00:02 - INFO -   - Benign: 1 (100.0%)
2025-12-16 18:00:02 - INFO - notepad.exe | Benign   | 0.1234
```

### Test cả thư mục + lưu CSV:

```powershell
python -m train.test_ember_model -d C:\samples --csv
```

**Kết quả:**
- Hiển thị kết quả trên console
- Lưu file CSV: `test_results_20251216_180000.csv`

### Xem chi tiết hướng dẫn test:

Xem file `train/TEST_MODEL.md` để biết chi tiết cách test model.

---

## 7. TROUBLESHOOTING

### ❌ Lỗi: `ModuleNotFoundError: No module named 'ember'`

**Nguyên nhân**: Không tìm thấy module `ember`

**Giải pháp**:
```powershell
# Đảm bảo chạy từ thư mục project root
cd D:\pbl6\ember
python -m train.ember_v1
```

### ❌ Lỗi: `Khong tim thay train_features_*.jsonl files`

**Nguyên nhân**: Dataset chưa có hoặc đường dẫn sai

**Giải pháp**:
```powershell
# Kiểm tra dataset
dir data\ember2018\train_features_*.jsonl

# Nếu không có, cần download dataset EMBER2018
# https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
```

### ❌ Lỗi: `File features KHONG co field 'label'!` (V1)

**Nguyên nhân**: Dataset không có field `label` (không chuẩn EMBER)

**Giải pháp**:
- **Option 1**: Sửa dataset, thêm field `label` vào mỗi record
- **Option 2**: Dùng **V2** thay vì V1 (V2 sẽ tạo label giả)

### ❌ Lỗi: `Memory error` hoặc `Out of memory`

**Nguyên nhân**: RAM không đủ

**Giải pháp**:
- **Option 1**: Tăng RAM (khuyến nghị 16GB+)
- **Option 2**: Dùng **V1** thay vì V2 (V1 tiết kiệm RAM hơn)
- **Option 3**: Đóng các ứng dụng khác để giải phóng RAM

### ❌ Lỗi: `LIEF version warning`

**Nguyên nhân**: Version LIEF không tương thích

**Giải pháp**:
```powershell
pip uninstall lief
pip install lief==0.12.3
```

### ❌ Lỗi: Training quá chậm

**Nguyên nhân**: CPU yếu hoặc dataset quá lớn

**Giải pháp**:
- Đảm bảo có **vectorized features** (`.dat` files) → V1 sẽ nhanh hơn
- Tăng số threads trong LightGBM (sửa trong code: `num_threads=8`)
- Đóng ứng dụng khác để dành CPU

---

## 📊 TÓM TẮT NHANH

### Train V1 (Chuẩn):
```powershell
cd D:\pbl6\ember
.\venv\Scripts\Activate.ps1
python -m train.ember_v1
```

### Train V2 (Linh hoạt):
```powershell
cd D:\pbl6\ember
.\venv\Scripts\Activate.ps1
python -m train.ember_v2
```

### Test Model:
```powershell
python -m train.test_ember_model -f file.exe
python -m train.test_ember_model -d folder --csv
```

---

## 📚 TÀI LIỆU THAM KHẢO

- **EMBER Paper**: https://arxiv.org/abs/1804.04637
- **EMBER Dataset**: https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
- **EMBER GitHub**: https://github.com/elastic/ember

---

**Chúc bạn train model thành công! 🎉**

