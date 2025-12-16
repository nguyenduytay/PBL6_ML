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

### 📂 Thư mục gốc `ember/`

```text
ember/
├── data/
│   └── ember2018/                 # Dataset EMBER2018 đã được trích xuất & vector hóa
│       ├── train_features_0.jsonl
│       ├── train_features_1.jsonl
│       ├── train_features_2.jsonl
│       ├── train_features_3.jsonl
│       ├── train_features_4.jsonl
│       ├── train_features_5.jsonl # 6 file features dùng để train
│       ├── test_features.jsonl    # Features dùng để test
│       ├── X_train.dat            # Vector hóa features train (numpy memmap)
│       ├── y_train.dat            # Nhãn train (0/1)
│       ├── X_test.dat             # Vector hóa features test
│       └── y_test.dat             # Nhãn test
│
├── ember/                         # Source code gốc của thư viện EMBER (clone từ GitHub)
│   ├── __init__.py                # Entry chính, định nghĩa API như create_vectorized_features, read_vectorized_features, predict_sample, ...
│   └── features.py                # Logic trích xuất features từ PE file
│
├── train/                         # Code train & test model trong dự án của bạn
│   ├── ember_v1.py                # Phiên bản trainer đầy đủ kiểm tra định dạng dataset, ưu tiên dùng vectorized features
│   ├── ember_v2.py                # Trainer đơn giản hơn, đọc trực tiếp JSONL và tự xây X, y
│   ├── test_ember_model.py        # Script test model đã train lên file đơn lẻ hoặc cả thư mục
│   ├── SETUP.md                   # Hướng dẫn cài đặt/chạy riêng cho thư mục train
│   └── TEST_MODEL.md              # Hướng dẫn chi tiết cách test model
│
├── scripts/                       # Script tiện ích đi kèm EMBER gốc
│   ├── classify_binaries.py       # Dùng model đã train để phân loại file PE từ dòng lệnh
│   └── init_ember.py              # Khởi tạo/tiền xử lý dataset EMBER (metadata, v.v.)
│
├── malconv/                       # Mô hình MalConv (CNN) thay thế/so sánh với EMBER-LightGBM
│   ├── malconv.py                 # Định nghĩa kiến trúc mạng MalConv
│   ├── multi_gpu.py               # Hỗ trợ train/infer trên nhiều GPU
│   ├── malconv.h5                 # Trọng số model MalConv đã train (Keras/HDF5)
│   └── README.md                  # Tài liệu riêng cho phần MalConv
│
├── resources/                     # Tài nguyên phục vụ demo & tài liệu
│   ├── ember-notebook.ipynb       # Notebook minh họa cách dùng EMBER
│   ├── ember2018-notebook.ipynb   # Notebook chuyên cho dataset EMBER2018
│   └── logo.png                   # Logo/ảnh minh họa
│
├── licenses/                      # License phụ thuộc
│   ├── AGPL-LICENSE-3.0.txt
│   └── MIT-LICENSE.txt
│
├── venv/                          # Virtualenv (môi trường ảo Python cho dự án)
│   ├── Scripts/                   # python.exe, pip.exe, activate, ...
│   └── Lib/site-packages/         # Các package: lightgbm, numpy, pandas, sklearn, ember, lief, ...
│
├── ember_model_pycharm.txt        # Model LightGBM đã train (export dạng text của LightGBM)
├── ember_training.log             # Log chi tiết quá trình training (từ ember_v1.py/ember_v2.py)
├── ember_test.log                 # Log quá trình test model (từ test_ember_model.py)
├── test_sample.exe                # File PE mẫu dùng để test nhanh model
├── requirements.txt               # Danh sách thư viện Python cần cài
├── LICENSE.txt                    # License chính của dự án
└── README.md                      # Tài liệu chính (file hiện tại)
```

### 📘 Giải thích chi tiết vai trò từng phần

- **`data/ember2018/`**  
  - **Chứa toàn bộ dataset EMBER2018** đã được tải về và xử lý:  
    - Các file `train_features_*.jsonl` và `test_features.jsonl` là **features dạng JSONL** do EMBER trích xuất từ file PE gốc.  
    - Các file `X_*.dat`, `y_*.dat` là **vector hóa** (matrix features và nhãn) ở dạng `numpy memmap` để train nhanh và tiết kiệm RAM.  
  - Các script train (`ember_v1.py`, `ember_v2.py`) sẽ **đọc dữ liệu từ đây**:
    - Nếu có `.dat` → đọc trực tiếp.
    - Nếu chưa có `.dat` → gọi `ember.create_vectorized_features(...)` để tạo.

- **`ember/`**  
  - Đây là **source gốc của thư viện EMBER** (clone từ `github.com/elastic/ember`).  
  - File chính:
    - `__init__.py`: định nghĩa API:
      - `create_vectorized_features(data_dir, feature_version=2)`
      - `read_vectorized_features(data_dir, feature_version=2)`
      - `predict_sample(model, file_or_bytes, feature_version=2)`
    - `features.py`: mô tả chi tiết cách trích xuất 2381 features từ file PE.
  - Các script train/test trong `train/` **import trực tiếp từ đây** để tạo features và dự đoán.

- **`train/`**  
  - Đây là **phần chính bạn đang dùng để train/test mô hình EMBER-LightGBM**:
    - **`ember_v1.py`**:
      - Kiểm tra format file `train_features_*.jsonl` rất kỹ (có `label`, đủ các field: `histogram`, `byteentropy`, `general`, `header`, `section`, `imports`, `exports`, `strings`, ...).  
      - Ưu tiên dùng các file `X_train.dat`, `y_train.dat`, `X_test.dat`, `y_test.dat` nếu đã có (tức là đã vector hóa).  
      - Nếu chưa có, sẽ **tự động vector hóa** từ JSONL sang `.dat` rồi mới train LightGBM.  
    - **`ember_v2.py`**:
      - Phiên bản đơn giản hơn, trong trường hợp dataset không chuẩn hoàn toàn như EMBER gốc:  
        - Cố gắng tạo vectorized features.  
        - Nếu lỗi, có logic fallback: đọc trực tiếp JSONL, tự build ma trận `X` và tạo nhãn giả để demo/training thử.  
      - Phù hợp khi bạn muốn **chạy thử/train demo** dù dataset chưa đúng chuẩn 100%.  
    - **`test_ember_model.py`**:
      - Dùng model đã train (`ember_model_pycharm.txt`) để:
        - Test **1 file** (`-f/--file`).
        - Test **cả thư mục** chứa nhiều file PE (`-d/--directory`, tùy chọn `--csv` để lưu kết quả).  
      - Tự kiểm tra file có phải PE (`MZ` header) trước khi phân tích.  
    - **`SETUP.md`** và **`TEST_MODEL.md`**:
      - Ghi chú/hướng dẫn chi tiết cách:
        - Chuẩn bị môi trường.
        - Chạy train.
        - Test model.

- **`scripts/`**  
  - **Script tiện ích** (mang tính “official” từ EMBER gốc):
    - `classify_binaries.py`:  
      - Dùng model đã train để phân loại file PE từ command line.  
      - Thường nhận model `-m`, rồi danh sách file hoặc thư mục.  
    - `init_ember.py`:  
      - Dùng để khởi tạo/tiền xử lý dataset (vd: sinh metadata, convert, v.v.) khi bạn làm việc trực tiếp với bộ dữ liệu thô của EMBER.

- **`malconv/`**  
  - Cung cấp **một pipeline khác** để phát hiện malware bằng **Deep Learning (MalConv - CNN)**:
    - `malconv.py`: định nghĩa kiến trúc mạng CNN đọc trực tiếp bytes của file.  
    - `multi_gpu.py`: hỗ trợ train/infer trên nhiều GPU.  
    - `malconv.h5`: file trọng số model MalConv đã train sẵn.  
  - Thư mục này **phục vụ so sánh**:
    - EMBER (LightGBM trên 2381 features) vs MalConv (CNN trên raw bytes).

- **`resources/`**  
  - Chứa **notebook minh họa**:
    - `ember-notebook.ipynb`, `ember2018-notebook.ipynb`:  
      - Giúp bạn xem code theo dạng notebook (Jupyter), thường dùng cho demo/giảng dạy.  
  - `logo.png`: logo dùng cho tài liệu / slide.

- **`licenses/` và `LICENSE.txt`**  
  - Thông tin về **bản quyền/phân phối** của dự án và các thư viện liên quan (AGPL, MIT, ...).

- **`venv/`**  
  - Môi trường ảo Python:
    - Đảm bảo các package như `lightgbm`, `numpy`, `pandas`, `scikit-learn`, `ember`, `lief`, ... được cài **cô lập** với hệ thống.  
  - Bạn dùng `venv\Scripts\activate` để kích hoạt trước khi chạy train/test.

- **Các file quan trọng ở root**:
  - **`ember_model_pycharm.txt`**:
    - File **model LightGBM đã train**, lưu theo định dạng text của LightGBM.  
    - Được tạo bởi các script train (`ember_v1.py`/`ember_v2.py`).  
    - Được load lại bằng:
      ```python
      import lightgbm as lgb
      model = lgb.Booster(model_file="ember_model_pycharm.txt")
      ```
  - **`ember_training.log`**:
    - Ghi chi tiết toàn bộ quá trình training:
      - Kiểm tra môi trường, dataset.
      - Tiến độ tạo vectorized features.
      - Thông số train, AUC/accuracy theo từng vòng.  
  - **`ember_test.log`**:
    - Log khi bạn chạy `test_ember_model.py` (kết quả dự đoán, thống kê malware/benign, top malware, v.v.).  
  - **`test_sample.exe`**:
    - File PE giả (chỉ có header tối thiểu) được script tạo ra để test nhanh model:
      - Dùng cho chức năng `test_sample` trong `ember_v1.py`, `ember_v2.py`, `test_ember_model.py`.  
  - **`requirements.txt`**:
    - Liệt kê các thư viện Python cần cài để chạy đầy đủ dự án (thường dùng với `pip install -r requirements.txt`).  
  - **`README.md`**:
    - File tài liệu chính mà bạn đang đọc, mô tả:
      - Mục tiêu dự án.
      - Cách cài đặt, train, test, tích hợp.
      - Cấu trúc dự án (phần vừa được cập nhật cho đúng với cấu trúc thật).

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

## 🔬 GIẢI THÍCH CHI TIẾT EMBER FEATURES

### 📊 Tổng quan về 8 nhóm features chính:

EMBER trích xuất **2381 features** từ file PE, được chia thành **8 nhóm chính**:

### 1. 📈 ByteHistogram (256 features)

**Mục đích**: Phân tích phân bố byte trong toàn bộ file PE.

**Cách hoạt động**:
- Đếm số lần xuất hiện của mỗi giá trị byte (0-255) trong file
- Tạo histogram 256 bins (mỗi bin = 1 giá trị byte)
- Chuẩn hóa thành xác suất (tổng = 1.0)

**Ví dụ**: 
- File có nhiều byte `0x00` → có thể là file đã pack/encrypt
- File có phân bố byte đều → entropy cao, có thể là mã hóa

**Code tham khảo**:
```python
# Trong ember/features.py
class ByteHistogram(FeatureType):
    dim = 256
    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()  # 256 giá trị
```

---

### 2. 🔄 ByteEntropyHistogram (256 features)

**Mục đích**: Phân tích entropy cục bộ kết hợp với giá trị byte (dựa trên nghiên cứu Saxe & Berlin, 2015).

**Cách hoạt động**:
- Chia file thành các **window** (2048 bytes) với **step** (1024 bytes)
- Với mỗi window:
  - Tính **entropy** (độ ngẫu nhiên) của window đó
  - Tạo histogram 16 bins cho entropy (0-8 bits)
  - Tạo histogram 16 bins cho byte values (0-255 → 16 bins)
- Kết hợp thành ma trận 16x16 = 256 features

**Tại sao quan trọng**:
- **Entropy thấp** + byte pattern đặc biệt → code thường (unpacked)
- **Entropy cao** → có thể là encrypted/packed/compressed data
- Malware thường pack code để tránh detection → entropy cao

**Code tham khảo**:
```python
class ByteEntropyHistogram(FeatureType):
    dim = 256
    def __init__(self, step=1024, window=2048):
        self.window = window  # Kích thước window
        self.step = step      # Bước nhảy
```

---

### 3. 📝 StringExtractor (104 features)

**Mục đích**: Trích xuất và phân tích chuỗi ký tự trong file PE.

**Các features**:
- `numstrings`: Tổng số chuỗi printable (ASCII 0x20-0x7F)
- `avlength`: Độ dài trung bình của chuỗi
- `printabledist`: Histogram phân bố 96 ký tự printable (0x20-0x7F)
- `entropy`: Entropy của chuỗi
- `paths`: Số lượng đường dẫn file (regex: `[c-zC-Z]:\\[^\\s]+`)
- `urls`: Số lượng URL (regex: `https?://[^\s]+`)
- `registry`: Số lượng registry keys (regex: `HKEY_[^\s]+`)
- `MZ`: Số lần xuất hiện chuỗi "MZ" (PE header signature)

**Tại sao quan trọng**:
- Malware thường chứa URL để download payload
- Registry keys để persistence
- File paths để tìm nạn nhân
- Nhiều chuỗi lạ → có thể là malware

**Code tham khảo**:
```python
class StringExtractor(FeatureType):
    dim = 104  # 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1
    def raw_features(self, bytez, lief_binary):
        # Tìm tất cả chuỗi printable
        allstrings = self._allstrings.findall(bytez)
        # Đếm paths, URLs, registry keys
        paths = len(self._paths.findall(bytez))
        urls = len(self._urls.findall(bytez))
        registry = len(self._registry.findall(bytez))
```

---

### 4. 📋 GeneralFileInfo (10 features)

**Mục đích**: Thông tin tổng quan về file PE.

**10 features**:
1. `size`: Kích thước file (bytes)
2. `vsize`: Virtual size (kích thước trong memory)
3. `has_debug`: Có thông tin debug không (0/1)
4. `exports`: Số lượng hàm export
5. `imports`: Số lượng hàm import
6. `has_relocations`: Có relocation table không (0/1)
7. `has_resources`: Có resource section không (0/1)
8. `has_signature`: Có digital signature không (0/1)
9. `has_tls`: Có Thread Local Storage không (0/1)
10. `symbols`: Số lượng symbols

**Tại sao quan trọng**:
- File **không có signature** → có thể là malware
- File **có TLS** → có thể là malware (thường dùng để anti-debug)
- File **không có debug info** → có thể đã bị strip (malware thường làm vậy)

---

### 5. 🏗️ HeaderFileInfo (62 features)

**Mục đích**: Thông tin chi tiết từ PE header (COFF + Optional Header).

**Các features**:
- **COFF Header**:
  - `timestamp`: Thời gian compile (có thể fake)
  - `machine`: Kiến trúc (x86/x64/ARM)
  - `characteristics`: Đặc điểm file (hashed thành 10 features)
- **Optional Header**:
  - `subsystem`: GUI/Console (hashed)
  - `dll_characteristics`: Đặc điểm DLL (hashed)
  - `magic`: PE32/PE32+ (hashed)
  - `major/minor_image_version`: Version của image
  - `major/minor_linker_version`: Version của linker
  - `major/minor_operating_system_version`: OS version
  - `major/minor_subsystem_version`: Subsystem version
  - `sizeof_code`: Kích thước code section
  - `sizeof_headers`: Kích thước headers
  - `sizeof_heap_commit`: Heap commit size

**Tại sao quan trọng**:
- **Linker version cũ** → có thể là malware cũ hoặc fake
- **Timestamp bất thường** → có thể bị fake
- **Subsystem** → GUI malware vs console malware

---

### 6. 📦 SectionInfo (255 features)

**Mục đích**: Thông tin chi tiết về các sections trong PE file.

**Các features**:
- **5 features tổng quan**:
  - Tổng số sections
  - Số sections có size = 0
  - Số sections có tên rỗng
  - Số sections RX (Read + Execute)
  - Số sections W (Write)
- **250 features hashed** (dùng FeatureHasher):
  - `section_sizes`: (tên section, size) → 50 features
  - `section_entropy`: (tên section, entropy) → 50 features
  - `section_vsize`: (tên section, virtual size) → 50 features
  - `entry_name`: Tên section chứa entry point → 50 features
  - `characteristics`: Đặc điểm của entry section → 50 features

**Tại sao quan trọng**:
- **Section tên lạ** (không phải .text, .data, .rdata) → có thể là malware
- **Section có entropy cao** → có thể đã pack/encrypt
- **Section có quyền Write + Execute** → rất đáng nghi (malware thường dùng để self-modify)

**Ví dụ tên section đáng nghi**:
- `.upx`, `.pack`, `.encrypted` → có thể đã pack
- `.text` có entropy cao → có thể đã encrypt

---

### 7. 📚 ImportsInfo (1280 features)

**Mục đích**: Phân tích các thư viện và hàm được import.

**Các features**:
- **256 features**: Libraries hashed (tên thư viện như `kernel32.dll`, `user32.dll`)
- **1024 features**: Imported functions hashed (tên đầy đủ như `kernel32.dll:CreateFileMappingA`)

**Cách hoạt động**:
- Dùng **FeatureHasher** để hash tên thư viện/hàm thành vector số
- Giữ nguyên thứ tự và tần suất xuất hiện

**Tại sao quan trọng**:
- **Hàm đáng nghi**: `VirtualAlloc`, `WriteProcessMemory`, `CreateRemoteThread` → có thể là malware
- **Thư viện đáng nghi**: `wininet.dll` (network), `advapi32.dll` (registry) → malware thường dùng
- **Ít imports** → có thể đã pack (unpack mới thấy imports thật)

**Ví dụ imports của malware**:
```python
# Malware thường import:
- kernel32.dll: VirtualAlloc, WriteProcessMemory, CreateRemoteThread
- wininet.dll: InternetOpen, InternetConnect, HttpSendRequest
- advapi32.dll: RegSetValueEx, RegCreateKey
```

---

### 8. 📤 ExportsInfo (128 features)

**Mục đích**: Phân tích các hàm được export (chủ yếu cho DLL).

**Các features**:
- **128 features**: Exported functions hashed

**Tại sao quan trọng**:
- **EXE file có exports** → đáng nghi (thường chỉ DLL mới export)
- **Tên export lạ** → có thể là malware
- **Nhiều exports** → có thể là legitimate DLL

---

### 9. 📂 DataDirectories (30 features)

**Mục đích**: Thông tin về 15 data directories đầu tiên trong PE.

**Các features**:
- Mỗi data directory có 2 features: `size` và `virtual_address`
- 15 directories × 2 = 30 features

**Data directories bao gồm**:
- Import Table, Export Table, Resource Table
- Exception Table, Certificate Table, Base Relocation Table
- Debug, Architecture, Global Pointer, TLS, Load Config
- Bound Import, IAT, Delay Import Descriptor, CLR Runtime Header

**Tại sao quan trọng**:
- **Certificate Table** → có digital signature (legitimate)
- **TLS Table** → có thể là malware (dùng để anti-debug)
- **CLR Runtime Header** → .NET executable

---

## 🎓 TẠI SAO DÙNG LIGHTGBM VÀ LƯU MODEL VÀO FILE .TXT?

### 🤔 Tại sao chọn LightGBM thay vì các thuật toán khác?

#### 1. **Hiệu suất vượt trội với dữ liệu tabular lớn**

**So sánh với các thuật toán khác**:

| Thuật toán | Ưu điểm | Nhược điểm | Phù hợp với EMBER? |
|-----------|---------|------------|-------------------|
| **LightGBM** ✅ | - Nhanh nhất (10-100x so với XGBoost)<br>- Chính xác cao<br>- Xử lý được dataset lớn (1M+ samples)<br>- Memory efficient<br>- Hỗ trợ GPU (tùy chọn) | - Cần tuning hyperparameters | ✅ **PHÙ HỢP NHẤT** |
| **XGBoost** | - Chính xác cao<br>- Robust | - Chậm hơn LightGBM 10-100x<br>- Tốn RAM hơn | ⚠️ Chậm với dataset lớn |
| **Random Forest** | - Dễ hiểu<br>- Không cần tuning nhiều | - Chậm với dataset lớn<br>- Tốn RAM<br>- Kém chính xác hơn | ❌ Không phù hợp |
| **Neural Network** | - Mạnh với dữ liệu phức tạp | - Cần GPU<br>- Training lâu<br>- Overfitting dễ xảy ra | ⚠️ Overkill cho tabular data |
| **SVM** | - Robust với noise | - Chậm với dataset lớn<br>- Không scale được | ❌ Không phù hợp |

**Với EMBER dataset (1M samples × 2381 features)**:
- LightGBM train trong **30-60 phút** trên CPU
- XGBoost cần **5-10 giờ**
- Random Forest có thể **không chạy được** (out of memory)

#### 2. **LightGBM được thiết kế cho tabular data**

**EMBER features là tabular data** (không phải image/text):
- 2381 features là **số liệu thống kê** (histogram, counts, hashed values)
- Không có **spatial/temporal patterns** như image/video
- LightGBM **tối ưu cho loại dữ liệu này**

**So sánh với Deep Learning**:
- **MalConv (CNN)**: Xử lý raw bytes như image → cần GPU, training lâu
- **LightGBM**: Xử lý features đã extract → nhanh hơn, chính xác tương đương

#### 3. **Memory efficiency**

**LightGBM dùng Gradient-based One-Side Sampling (GOSS)**:
- Chỉ train trên **một phần samples** quan trọng nhất
- Giảm memory usage **10-100x** so với XGBoost
- Với dataset 1M samples:
  - LightGBM: ~8-16GB RAM
  - XGBoost: ~32-64GB RAM (có thể out of memory)

#### 4. **Tốc độ inference nhanh**

**Sau khi train, model cần predict nhanh**:
- LightGBM: **Vài giây/file** (CPU)
- MalConv: **Vài chục giây/file** (cần GPU)
- Phù hợp cho **production environment** (real-time scanning)

#### 5. **Interpretability (khả năng giải thích)**

**LightGBM có thể xem feature importance**:
```python
# Xem features quan trọng nhất
feature_importance = model.feature_importance()
# Có thể biết malware thường có features gì
```

**Deep Learning (MalConv)**: Black box, khó giải thích

---

### 📄 Tại sao lưu model vào file .TXT thay vì .PKL, .H5, .BIN?

#### 1. **LightGBM hỗ trợ nhiều format, nhưng .TXT là lựa chọn tốt nhất**

**Các format LightGBM hỗ trợ**:

| Format | Ưu điểm | Nhược điểm | Khi nào dùng? |
|--------|---------|------------|---------------|
| **.txt** ✅ | - **Human-readable** (có thể mở bằng text editor)<br>- **Debug dễ dàng** (xem cấu trúc cây)<br>- **Portable** (không phụ thuộc Python version)<br>- **Nhẹ** (text format)<br>- **Version-independent** | - File lớn hơn binary một chút | ✅ **KHUYẾN NGHỊ** cho development |
| **.bin** | - Nhỏ hơn .txt<br>- Load nhanh hơn | - Không đọc được<br>- Phụ thuộc LightGBM version | Production (nếu cần tối ưu) |
| **.pkl** | - Python standard | - Phụ thuộc Python version<br>- Không portable | Không khuyến nghị |
| **.h5** | - HDF5 format | - Cần thư viện HDF5<br>- Phức tạp hơn | Không khuyến nghị |

#### 2. **.TXT format cho phép debug và hiểu model**

**Ví dụ nội dung file .txt**:
```
tree=0
num_leaves=31
split_feature=0 5 9 12 ...
split_gain=0.5 0.3 0.2 ...
threshold=0.5 0.3 0.2 ...
left_child=1 3 5 ...
right_child=2 4 6 ...
leaf_value=0.1 0.9 0.2 ...
```

**Có thể**:
- Xem cấu trúc cây quyết định
- Hiểu tại sao model predict như vậy
- Debug khi model hoạt động sai

**Với .bin/.pkl**: Không thể đọc được, phải dùng LightGBM API

#### 3. **Portability (tính di động)**

**File .txt**:
- Có thể copy sang máy khác (Windows/Linux/Mac)
- Không phụ thuộc Python version
- Không phụ thuộc LightGBM version (miễn là cùng major version)

**File .pkl**:
- Phụ thuộc Python version (Python 3.8 .pkl không load được trên Python 3.10)
- Phụ thuộc thư viện (nếu dùng pickle protocol mới)

#### 4. **File size không phải vấn đề lớn**

**So sánh kích thước**:
- Model EMBER: ~10-20MB (.txt) vs ~8-15MB (.bin)
- Chênh lệch **không đáng kể** (chỉ vài MB)
- **Ưu điểm của .txt** (debug, portable) **lớn hơn** nhược điểm (file lớn hơn một chút)

#### 5. **LightGBM khuyến nghị dùng .txt cho development**

**Theo tài liệu LightGBM**:
- `.txt` format được khuyến nghị cho **development/debugging**
- `.bin` format được khuyến nghị cho **production** (nếu cần tối ưu)

**Trong dự án này**:
- Đang ở giai đoạn **development/research**
- Cần **debug và hiểu model**
- → **.txt là lựa chọn đúng**

---

## 🚀 HƯỚNG DẪN TRAIN LIGHTGBM VÀ LƯU MODEL .TXT

### Bước 1: Chuẩn bị dữ liệu

```python
import ember
import numpy as np

# 1. Tạo vectorized features từ JSONL files
data_dir = "data/ember2018"
ember.create_vectorized_features(data_dir, feature_version=2)

# 2. Load vectorized features (memory-mapped, tiết kiệm RAM)
X_train, y_train, X_test, y_test = ember.read_vectorized_features(
    data_dir, feature_version=2
)

print(f"Train: {X_train.shape}")  # (800000, 2381)
print(f"Test: {X_test.shape}")      # (200000, 2381)
```

### Bước 2: Cấu hình LightGBM parameters

```python
import lightgbm as lgb

params = {
    'objective': 'binary',           # Binary classification (malware/benign)
    'metric': 'auc',                  # Metric để đánh giá (Area Under Curve)
    'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree
    'num_leaves': 31,                 # Số leaves trong mỗi cây (nhỏ hơn = ít overfit)
    'learning_rate': 0.05,            # Learning rate (nhỏ = train chậm nhưng chính xác hơn)
    'feature_fraction': 0.9,           # Dùng 90% features mỗi lần (tránh overfit)
    'bagging_fraction': 0.8,          # Dùng 80% samples mỗi lần (tránh overfit)
    'bagging_freq': 5,                # Bagging mỗi 5 iterations
    'verbose': 0,                     # Không in log chi tiết
    'num_threads': 4,                 # Số threads (tùy CPU)
    'force_col_wise': True            # Tối ưu cho dataset lớn (column-wise)
}
```

**Giải thích các parameters quan trọng**:
- **`num_leaves`**: Số lá trong mỗi cây. Nhỏ hơn = ít overfit nhưng có thể underfit
- **`learning_rate`**: Tốc độ học. Nhỏ hơn = chính xác hơn nhưng train lâu hơn
- **`feature_fraction`**: Random Forest-like, dùng một phần features → giảm overfit
- **`bagging_fraction`**: Dùng một phần samples → giảm overfit

### Bước 3: Training với early stopping

```python
# Tạo LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Training với callbacks
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],           # Validation set để early stopping
    num_boost_round=1000,              # Tối đa 1000 cây
    callbacks=[
        lgb.early_stopping(50),        # Dừng nếu không cải thiện sau 50 rounds
        lgb.log_evaluation(100)        # In log mỗi 100 rounds
    ]
)
```

**Early stopping**:
- Nếu validation AUC không cải thiện sau 50 rounds → dừng
- Tránh overfitting
- Tiết kiệm thời gian

### Bước 4: Lưu model vào file .TXT

```python
# Lưu model
model_path = "ember_model_pycharm.txt"
model.save_model(str(model_path))
print(f"Model đã được lưu: {model_path}")
```

**LightGBM tự động chọn format dựa trên extension**:
- `.txt` → Text format (human-readable)
- `.bin` → Binary format (nhỏ hơn, nhanh hơn)

### Bước 5: Load và sử dụng model

```python
# Load model từ file .txt
loaded_model = lgb.Booster(model_file="ember_model_pycharm.txt")

# Predict
y_pred = loaded_model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Threshold 0.5

# Đánh giá
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(y_test, y_pred_binary)
auc = roc_auc_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
```

---

## ⚖️ SO SÁNH EMBER vs MALCONV vs DYNAMIC ANALYSIS

### 📊 Bảng so sánh tổng quan:

| Tiêu chí | **EMBER (LightGBM)** | **MalConv (CNN)** | **Dynamic Analysis** |
|----------|---------------------|-------------------|---------------------|
| **Phương pháp** | Static analysis + ML (LightGBM) | Static analysis + Deep Learning (CNN) | Chạy malware trong sandbox |
| **Input** | 2381 features từ PE structure | Raw bytes (1MB đầu tiên) | File PE thực thi |
| **Tốc độ** | ⚡ **Vài giây/file** | 🐢 Vài chục giây/file | 🐌 Vài phút/file |
| **Độ chính xác** | ✅ **>95%** | ✅ >95% | ✅ **>98%** (cao nhất) |
| **Cần GPU?** | ❌ Không | ✅ Cần | ❌ Không |
| **RAM cần** | 💾 8-16GB | 💾 16-32GB | 💾 4-8GB |
| **Training time** | ⏱️ 30-60 phút | ⏱️ Vài giờ (với GPU) | N/A (không train) |
| **Interpretability** | ✅ Có (feature importance) | ❌ Không (black box) | ✅ Có (xem hành vi) |
| **Phát hiện obfuscation?** | ⚠️ Một phần (qua entropy) | ✅ Tốt (raw bytes) | ✅ **Tốt nhất** (thấy hành vi thật) |
| **Phát hiện zero-day?** | ⚠️ Một phần | ⚠️ Một phần | ✅ **Tốt nhất** |
| **False Positive Rate** | ✅ <1% | ⚠️ 2-5% | ✅ <0.5% |
| **Production ready?** | ✅ **Có** | ⚠️ Cần GPU | ❌ Quá chậm |

---

### 🔍 Giải thích chi tiết từng phương pháp:

#### 1. **EMBER (LightGBM) - Static Analysis với Hand-crafted Features**

**Cách hoạt động**:
1. Parse PE file → trích xuất 2381 features (headers, sections, imports, strings...)
2. Train LightGBM trên 1M samples
3. Predict: File mới → extract features → predict

**Ưu điểm**:
- ✅ **Nhanh nhất** (vài giây/file)
- ✅ **Không cần GPU**
- ✅ **Interpretable** (biết features nào quan trọng)
- ✅ **Memory efficient** (8-16GB RAM)
- ✅ **Production ready** (có thể scale lên hàng nghìn file/phút)

**Nhược điểm**:
- ❌ **Phụ thuộc vào features** (nếu malware obfuscate tốt → features bị sai)
- ❌ **Không phát hiện được logic phức tạp** (chỉ dựa vào structure)
- ❌ **Có thể bị bypass** nếu attacker biết features nào được dùng

**Khi nào dùng**:
- ✅ **Production environment** (real-time scanning)
- ✅ **Large-scale scanning** (hàng triệu file)
- ✅ **Khi cần tốc độ** (email gateway, file upload)

---

#### 2. **MalConv (CNN) - Static Analysis với Deep Learning**

**Cách hoạt động**:
1. Lấy **1MB đầu tiên** của file PE (như image 1D)
2. Dùng **CNN** để học patterns từ raw bytes
3. Train trên GPU (vài giờ)
4. Predict: File mới → CNN → predict

**Ưu điểm**:
- ✅ **Tự động học features** (không cần hand-craft)
- ✅ **Phát hiện patterns phức tạp** (CNN mạnh với patterns)
- ✅ **Không phụ thuộc PE structure** (xử lý raw bytes)

**Nhược điểm**:
- ❌ **Cần GPU** để train (vài giờ) và inference (chậm trên CPU)
- ❌ **Black box** (không biết tại sao predict như vậy)
- ❌ **Tốn RAM** (16-32GB)
- ❌ **Chậm hơn EMBER** (vài chục giây/file trên CPU)
- ❌ **Chỉ xử lý 1MB đầu** (có thể miss payload ở cuối file)

**Khi nào dùng**:
- ✅ **Research** (thử nghiệm deep learning)
- ✅ **Khi có GPU** và không cần tốc độ
- ✅ **Khi muốn tự động học features** (không muốn hand-craft)

---

#### 3. **Dynamic Analysis - Chạy malware trong sandbox**

**Cách hoạt động**:
1. Chạy file PE trong **sandbox** (môi trường cách ly)
2. Monitor **hành vi** (API calls, network, file system, registry)
3. Phân tích hành vi → quyết định malware/benign

**Ưu điểm**:
- ✅ **Chính xác nhất** (>98%) - thấy hành vi thật
- ✅ **Phát hiện zero-day** tốt (không cần train)
- ✅ **Phát hiện obfuscation** (dù code bị pack, hành vi vẫn thấy)
- ✅ **Interpretable** (biết malware làm gì)

**Nhược điểm**:
- ❌ **Chậm nhất** (vài phút/file - phải chạy thực tế)
- ❌ **Không scale được** (không thể scan hàng nghìn file/phút)
- ❌ **Có thể bị anti-VM** (malware phát hiện sandbox → không chạy)
- ❌ **Tốn tài nguyên** (cần nhiều VM/sandbox)

**Khi nào dùng**:
- ✅ **Deep analysis** (phân tích sâu một vài file đáng nghi)
- ✅ **Threat intelligence** (hiểu malware làm gì)
- ✅ **Khi cần độ chính xác cao nhất** (không cần tốc độ)

---

### 🎯 Kết luận: Khi nào dùng phương pháp nào?

**Trong thực tế, người ta thường dùng KẾT HỢP**:

1. **Bước 1 - EMBER (LightGBM)**: 
   - Scan **hàng triệu file** nhanh chóng
   - Lọc ra **các file đáng nghi** (score > 0.7)

2. **Bước 2 - MalConv (nếu có GPU)**:
   - Phân tích sâu hơn các file đáng nghi
   - Double-check với deep learning

3. **Bước 3 - Dynamic Analysis**:
   - Chỉ chạy với **file rất đáng nghi** (score > 0.9)
   - Hiểu malware làm gì (threat intelligence)

**Ví dụ pipeline thực tế**:
```
Email attachment → EMBER scan (2 giây) 
  → Nếu score > 0.7: MalConv scan (30 giây)
    → Nếu score > 0.9: Dynamic analysis (5 phút)
      → Quyết định cuối cùng
```

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
