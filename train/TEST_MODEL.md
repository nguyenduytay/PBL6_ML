# Hướng Dẫn Test EMBER Model

Script `test_ember_model.py` dùng để test model EMBER đã train với các file PE thực tế.
PE = Portable Executable = định dạng file thực thi của Windows

## 📋 Yêu Cầu

- Python 3.8+
- Model đã train: `ember_model_pycharm.txt`
- Các dependencies: `lightgbm`, `ember` (đã cài khi train)

## 🚀 Cách Sử Dụng

### 1. Test Một File Đơn

```bash
python -m train.test_ember_model -m ember_model_pycharm.txt -f path/to/file.exe
```

**Ví dụ:**

```bash
python -m train.test_ember_model -f C:\Windows\System32\notepad.exe
```

### 2. Test Cả Thư Mục

```bash
python -m train.test_ember_model -m ember_model_pycharm.txt -d path/to/directory
```

**Ví dụ:**

```bash
python -m train.test_ember_model -d C:\samples
```

### 3. Test Và Lưu Kết Quả CSV

```bash
python -m train.test_ember_model -d C:\samples --csv
```

File CSV sẽ được lưu với tên: `test_results_YYYYMMDD_HHMMSS.csv`

### 4. Test File Mẫu Tự Động

```bash
python -m train.test_ember_model --sample
```

## 📊 Kết Quả

### Output Console:

```
2025-11-01 10:00:00 - INFO - Model path: D:\pbl6\ember\ember_model_pycharm.txt
2025-11-01 10:00:01 - INFO - ✓ Model đã load thành công
2025-11-01 10:00:01 - INFO -   - Số cây: 500
2025-11-01 10:00:01 - INFO -   - Số features: 2,381
2025-11-01 10:00:01 - INFO - Đang phân tích: sample.exe
2025-11-01 10:00:02 - INFO - ================================================================================
2025-11-01 10:00:02 - INFO - KẾT QUẢ PHÂN TÍCH
2025-11-01 10:00:02 - INFO - ================================================================================
2025-11-01 10:00:02 - INFO - Tổng số file: 1
2025-11-01 10:00:02 - INFO -   - Malware: 0 (0.0%)
2025-11-01 10:00:02 - INFO -   - Benign: 1 (100.0%)
2025-11-01 10:00:02 - INFO - ================================================================================
```

### Giải Thích Score:

- **Score**: Giá trị từ 0.0 đến 1.0
  - `score > 0.5`: **Malware** (xác suất cao là mã độc)
  - `score <= 0.5`: **Benign** (an toàn)
- **Score càng gần 1.0**: Xác suất malware càng cao
- **Score càng gần 0.0**: Xác suất benign càng cao

## 📝 Các Tùy Chọn

| Tùy chọn                | Mô tả                         | Ví dụ                        |
| ----------------------- | ----------------------------- | ---------------------------- |
| `-m, --model`           | Đường dẫn model               | `-m ember_model_pycharm.txt` |
| `-f, --file`            | Test một file                 | `-f sample.exe`              |
| `-d, --directory`       | Test cả thư mục               | `-d C:\samples`              |
| `-v, --feature-version` | Feature version (mặc định: 2) | `-v 2`                       |
| `--csv`                 | Lưu kết quả CSV               | `--csv`                      |
| `--sample`              | Test với file mẫu             | `--sample`                   |

## 💡 Ví Dụ Nâng Cao

### Test nhiều file và export CSV:

```bash
python -m train.test_ember_model -d C:\malware_samples --csv
```

### Test với model khác:

```bash
python -m train.test_ember_model -m data/ember2018/ember_model_2018.txt -f test.exe
```

### Test chỉ file .exe:

Script tự động quét các file: `.exe`, `.dll`, `.sys`, `.scr`, `.com`, `.bat`, `.cmd`

## 📁 Các File Có Thể Test

### ✅ **File PE (Portable Executable) - Test Được**

EMBER chỉ có thể test **file PE** vì model được train trên dataset PE và features được extract từ cấu trúc PE.

#### Các loại file PE được hỗ trợ:

| Extension | Mô tả                  | Ví dụ                        |
| --------- | ---------------------- | ---------------------------- |
| `.exe`    | Executable files       | `notepad.exe`, `chrome.exe`  |
| `.dll`    | Dynamic Link Library   | `kernel32.dll`, `user32.dll` |
| `.sys`    | System drivers         | `ntfs.sys`, `disk.sys`       |
| `.scr`    | Screen savers          | `mystyle.scr`                |
| `.com`    | Command files          | `command.com`                |
| `.ocx`    | ActiveX controls       | `mscomctl.ocx`               |
| `.cpl`    | Control Panel applets  | `appwiz.cpl`                 |
| `.drv`    | Device drivers         | `vga.drv`                    |
| `.efi`    | EFI executables        | `bootx64.efi`                |
| `.bat`    | Batch files (đơn giản) | `setup.bat`                  |
| `.cmd`    | Command scripts        | `install.cmd`                |

#### Yêu cầu file PE:

- ✅ Phải có **MZ header** ở đầu file (signature `4D 5A`)
- ✅ Phải có **PE signature** (`PE\0\0`)
- ✅ Phải có **PE structure** hợp lệ (sections, imports, exports...)

### ❌ **File Không Phải PE - Không Test Được**

Các file sau **KHÔNG** phải PE và EMBER **KHÔNG THỂ** test:

| Loại File            | Extension                                        | Lý Do                                   |
| -------------------- | ------------------------------------------------ | --------------------------------------- |
| **Text files**       | `.txt`, `.log`, `.csv`, `.json`, `.xml`, `.yaml` | Không có PE structure                   |
| **Office documents** | `.docx`, `.xlsx`, `.pptx`, `.pdf`                | ZIP archive hoặc binary format khác     |
| **Archives**         | `.zip`, `.rar`, `.7z`, `.tar`, `.gz`             | Compressed files, không phải executable |
| **Images**           | `.jpg`, `.png`, `.gif`, `.bmp`, `.svg`           | Image formats                           |
| **Media**            | `.mp3`, `.mp4`, `.avi`, `.mkv`                   | Media formats                           |
| **Scripts**          | `.py`, `.js`, `.vbs`, `.ps1`                     | Script files (không phải PE)            |
| **Java**             | `.jar`, `.class`                                 | Java bytecode, không phải PE            |
| **Android**          | `.apk`                                           | Android package, không phải PE          |
| **Linux**            | `.elf`, `.so`, `.bin`                            | Linux executables, không phải PE        |

### 🔍 Kiểm Tra File Có Phải PE Không?

#### Cách 1: Dùng Script (Tự Động)

Script `test_ember_model.py` **tự động kiểm tra** file có phải PE trước khi test:

- ✅ File PE → Tiến hành phân tích
- ❌ Không phải PE → Cảnh báo và bỏ qua

#### Cách 2: Kiểm Tra Thủ Công

**Windows PowerShell:**

```powershell
# Đọc 2 bytes đầu file
$bytes = [System.IO.File]::ReadAllBytes("file.exe")
if ($bytes[0] -eq 0x4D -and $bytes[1] -eq 0x5A) {
    Write-Host "Đây là file PE"
} else {
    Write-Host "Không phải file PE"
}
```

**Python:**

```python
with open('file.exe', 'rb') as f:
    header = f.read(2)
    if header == b'MZ':
        print("Đây là file PE")
    else:
        print("Không phải file PE")
```

**Command Line (Windows):**

```cmd
certutil -hashfile file.exe SHA256
REM Nếu file PE hợp lệ, sẽ hiển thị hash
```

### 💡 Ví Dụ Cụ Thể

#### ✅ Test Được:

```powershell
# File .exe
python -m train.test_ember_model -f C:\Windows\System32\notepad.exe

# File .dll
python -m train.test_ember_model -f C:\Windows\System32\kernel32.dll

# File .sys
python -m train.test_ember_model -f C:\Windows\System32\drivers\ntfs.sys
```

#### ❌ Không Test Được:

```powershell
# File .log - Sẽ báo lỗi
python -m train.test_ember_model -f log.txt
# Output: ⚠️  File 'log.txt' không phải file PE hợp lệ!

# File .pdf - Sẽ báo lỗi
python -m train.test_ember_model -f document.pdf
# Output: ⚠️  File 'document.pdf' không phải file PE hợp lệ!
```

### 🎯 Tại Sao Chỉ PE Files?

1. **Model được train trên PE**: Dataset EMBER2018 chỉ chứa file PE
2. **Features từ PE structure**: EMBER extract:
   - Sections (`.text`, `.data`, `.rdata`)
   - Imports (APIs như `kernel32.dll`)
   - Exports (hàm export)
   - PE Headers (machine type, architecture)
3. **LIEF library**: Chỉ parse định dạng PE

→ Muốn test file khác, cần:

- Model khác được train cho format đó
- Feature extractor khác
- Framework khác (không phải EMBER)

## ⚠️ Lưu Ý

1. **File size**: Script có thể test file PE lớn, nhưng sẽ tốn thời gian
2. **Memory**: Model cần ~200MB RAM khi load
3. **False positives**: Một số file hợp pháp có thể bị nhận diện sai (score 0.5-0.7)
4. **Log file**: Kết quả được log vào `ember_test.log`

## 🔍 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'ember'`

**Giải pháp**: Đảm bảo đang chạy từ thư mục project root:

```bash
cd D:\pbl6\ember
python -m train.test_ember_model -f test.exe
```

### Lỗi: `Model file không tồn tại`

**Giải pháp**: Kiểm tra đường dẫn model:

```bash
# Dùng đường dẫn tuyệt đối
python -m train.test_ember_model -m D:\pbl6\ember\ember_model_pycharm.txt -f test.exe
```

### Lỗi: `LIEF version warning`

Cảnh báo này không ảnh hưởng đến kết quả, nhưng có thể có sự khác biệt nhỏ so với training.

### Lỗi: `File không phải file PE hợp lệ!`

**Nguyên nhân**: File bạn đang test không phải file PE (Portable Executable).

**Giải pháp**:

1. Kiểm tra extension file:

   - ✅ Test được: `.exe`, `.dll`, `.sys`, `.scr`, `.com`, `.ocx`, `.cpl`
   - ❌ Không test được: `.log`, `.txt`, `.pdf`, `.zip`, `.jpg`, v.v.

2. Kiểm tra MZ header:

   ```powershell
   # Đọc 2 bytes đầu file
   Format-Hex -Path "file.exe" -Count 2
   # Phải thấy: 4D 5A (MZ)
   ```

3. Chỉ test file PE:
   ```powershell
   # Ví dụ đúng
   python -m train.test_ember_model -f C:\Windows\System32\notepad.exe
   ```

**Lưu ý**: EMBER **CHỈ** test được file PE. Xem phần [📁 Các File Có Thể Test](#-các-file-có-thể-test) để biết chi tiết.

## 📈 Hiệu Suất

- **Tốc độ**: ~1-3 giây/file (tùy kích thước)
- **Độ chính xác**: Tương đương với kết quả training
  - Accuracy: ~94%
  - Precision: ~98%
  - Recall: ~90%
