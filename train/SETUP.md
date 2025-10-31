## Hướng dẫn chạy EMBER Trainer (train/ember_pycharm.py)

### 1) Yêu cầu hệ thống

- Python 3.10 (đã có sẵn trên máy bạn)
- Windows PowerShell
- RAM tối thiểu 8GB (khuyến nghị 16GB)
- Dataset EMBER2018 đã giải nén tại `data/ember2018/`

Thư mục dataset nên có các file lớn sau (đã có sẵn theo bạn cung cấp):

- `data/ember2018/train_features_0.jsonl` … `train_features_5.jsonl`
- `data/ember2018/test_features.jsonl`
- (Tùy chọn) `data/ember2018/X_train.dat`, `y_train.dat` nếu đã vectorize trước đó

### 2) Tạo và kích hoạt môi trường ảo (Python 3.10)

Chạy các lệnh trong PowerShell tại thư mục dự án `d:\pbl6\ember`:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 3) Cài dependencies

- Cài các gói chính (tương thích Python 3.10):

```powershell
pip install tqdm numpy pandas lightgbm scikit-learn lief==0.12.3 psutil
```

Lưu ý: Dùng `lief==0.12.3` để tương thích tốt với Python 3.10. (Phiên bản 0.9.0 không có wheel cho Py3.10/3.12.)

### 4) Cấu trúc chạy và đường dẫn

Script training nằm tại: `train/ember_pycharm.py`

Script sẽ tự thêm đường dẫn `ember/` vào `sys.path` để import `ember` từ source trong project.

Dataset mặc định đọc tại: `data/ember2018/` (cùng cấp với thư mục `ember/`).

### 5) Chạy training

Chạy bằng module để đảm bảo relative import:

```powershell
python -m train.ember_pycharm
```

Sau khi chạy:

- Model được lưu tại: `train/ember_model_pycharm.txt`
- Log chi tiết tại: `train/ember_training.log`

### 6) Ghi chú hiệu năng

- Dataset rất lớn (vài GB). Training có thể mất 30–60 phút tùy cấu hình.
- Nên đóng bớt ứng dụng khác để dành RAM/CPU cho quá trình huấn luyện.

### 7) Sử dụng model đã train

- Load model LightGBM và dùng trực tiếp:

```python
import lightgbm as lgb
model = lgb.Booster(model_file="train/ember_model_pycharm.txt")

# Ví dụ: dự đoán xác suất malware cho ma trận đặc trưng X (numpy array)
# y_prob = model.predict(X)
```

- Chấm điểm trực tiếp một file PE (.exe/.dll) bằng EMBER (tiện cho kiểm thử nhanh):

```python
import ember
import lightgbm as lgb

model = lgb.Booster(model_file="train/ember_model_pycharm.txt")
score = ember.predict_sample(model, r"C:\\path\\to\\file.exe", feature_version=2)
print("Malware score:", score)
print("Prediction:", "Malware" if score > 0.5 else "Benign")
```

- Chấm điểm hàng loạt file trong một thư mục (ví dụ đơn giản):

```python
import os
import ember
import lightgbm as lgb

model = lgb.Booster(model_file="train/ember_model_pycharm.txt")
folder = r"C:\\samples"

for name in os.listdir(folder):
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        try:
            score = ember.predict_sample(model, path, feature_version=2)
            print(name, score, "Malware" if score > 0.5 else "Benign")
        except Exception as e:
            print("Skip", name, e)
```

### 8) Troubleshooting

- Lỗi không cài được LIEF trên Python 3.12:

  - Dùng môi trường ảo Python 3.10 như hướng dẫn trên.
  - Đảm bảo cài đúng: `pip install lief==0.12.3`

- Lỗi `ModuleNotFoundError: ember` khi chạy:

  - Hãy chạy từ thư mục project và dùng lệnh: `python -m train.ember_pycharm`
  - Đảm bảo thư mục `ember/` (source) nằm cạnh thư mục `train/` trong dự án.

- Thiếu RAM/Training chậm:
  - Giảm ứng dụng chạy nền, hoặc tăng RAM ảo (pagefile) trên Windows.

### 9) Lệnh nhanh (tóm tắt)

```powershell
cd d:\pbl6\ember
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install tqdm numpy pandas lightgbm scikit-learn lief==0.12.3 psutil
python -m train.ember_pycharm
```
