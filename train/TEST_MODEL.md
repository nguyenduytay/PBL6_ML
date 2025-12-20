# Hướng Dẫn Test EMBER Model

Script `test_ember_model.py` dùng để test model EMBER đã train với các file PE thực tế.
PE = Portable Executable = định dạng file thực thi của Windows

## 📋 Yêu Cầu

- Python 3.8+
- Model đã train: `ember_model_pycharm.txt`
- Các dependencies: `lightgbm`, `ember` (đã cài khi train)
- **Hệ điều hành**: Windows, Linux (Ubuntu), hoặc macOS (script chạy được trên mọi OS, nhưng chỉ test được file PE)

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

## 🐧 Test Trên Ubuntu/Linux

### ⚠️ Vấn Đề Chính

**EMBER chỉ test được file PE (Windows executables)**, không test được file Linux (ELF format).

Trên Ubuntu/Linux:
- ❌ **KHÔNG** có file PE mặc định (vì Linux dùng ELF format)
- ❌ **KHÔNG** thể test file Linux executables (`.elf`, `.so`, `.bin`)
- ✅ **CÓ THỂ** test file PE nếu bạn có file PE (ví dụ: file từ Windows)

### ✅ Giải Pháp: Test File PE Trên Ubuntu

#### Cách 1: Copy File PE Từ Windows

1. **Copy file PE từ Windows sang Ubuntu:**
   ```bash
   # Ví dụ: Copy file từ Windows sang Ubuntu qua SCP
   scp user@windows:/path/to/file.exe /home/user/samples/
   
   # Hoặc dùng USB, network share, v.v.
   ```

2. **Test file PE trên Ubuntu:**
   ```bash
   cd /path/to/ember
   python -m train.test_ember_model -m ember_model_pycharm.txt -f /home/user/samples/file.exe
   ```

#### Cách 2: Download File PE Mẫu

1. **Download file PE mẫu từ internet:**
   ```bash
   # Ví dụ: Download notepad.exe từ Windows ISO hoặc sample repository
   wget https://example.com/samples/test.exe -O /tmp/test.exe
   ```

2. **Test file đã download:**
   ```bash
   python -m train.test_ember_model -f /tmp/test.exe
   ```

#### Cách 3: Tạo File PE Test Đơn Giản

1. **Tạo file PE test đơn giản:**
   ```python
   # create_test_pe.py
   pe_header = b'MZ' + b'\x00' * 58 + b'PE\x00\x00' + b'\x00' * 1000
   with open('test_sample.exe', 'wb') as f:
       f.write(pe_header)
   ```

2. **Chạy script tạo file:**
   ```bash
   python create_test_pe.py
   python -m train.test_ember_model -f test_sample.exe
   ```

### ❌ Tại Sao Không Test Được File Linux?

**File Linux (ELF format) KHÔNG phải PE format:**

| Format | OS | Header | EMBER Test? |
|--------|-----|--------|-------------|
| **PE** | Windows | `MZ` + `PE\x00\x00` | ✅ Có |
| **ELF** | Linux | `\x7F ELF` | ❌ Không |
| **Mach-O** | macOS | `FE ED FA CE` | ❌ Không |

**Ví dụ kiểm tra file Linux:**
```bash
# Kiểm tra file Linux executable
file /usr/bin/ls
# Output: /usr/bin/ls: ELF 64-bit LSB executable, x86-64...

# Kiểm tra header
hexdump -C /usr/bin/ls | head -1
# Output: 00000000  7f 45 4c 46 02 01 01 00  ...ELF....

# → File này KHÔNG phải PE, EMBER KHÔNG test được!
```

### 💡 Ví Dụ Test Trên Ubuntu

```bash
# 1. Kiểm tra file có phải PE không
file sample.exe
# Output: sample.exe: PE32 executable (console) Intel 80386...

# 2. Test file PE
cd /path/to/ember
python -m train.test_ember_model -m ember_model_pycharm.txt -f /path/to/sample.exe

# 3. Test nhiều file PE trong thư mục
python -m train.test_ember_model -d /home/user/pe_samples --csv
```

### 🔍 Troubleshooting Trên Ubuntu

#### Lỗi: `File không phải file PE hợp lệ!`

**Nguyên nhân**: Bạn đang cố test file Linux (ELF) thay vì file PE.

**Giải pháp**:
```bash
# Kiểm tra file format
file your_file
# Nếu thấy "ELF" → Đây là file Linux, KHÔNG test được
# Nếu thấy "PE32" → Đây là file PE, test được

# Chỉ test file PE
python -m train.test_ember_model -f file_pe.exe  # ✅ Đúng
python -m train.test_ember_model -f /usr/bin/ls    # ❌ Sai (file Linux)
```

#### Lỗi: `ModuleNotFoundError: No module named 'ember'`

**Giải pháp**:
```bash
# Đảm bảo đang ở project root
cd /path/to/ember

# Kiểm tra PYTHONPATH
export PYTHONPATH=/path/to/ember:$PYTHONPATH

# Chạy lại
python -m train.test_ember_model -f sample.exe
```

#### Lỗi: `LIEF version warning`

Cảnh báo này không ảnh hưởng, nhưng có thể cài đúng version:
```bash
pip install lief==0.9.0
```

### 📝 Tóm Tắt

| Tình Huống | Test Được? | Giải Pháp |
|------------|-------------|-----------|
| File PE trên Ubuntu | ✅ Có | Copy file PE từ Windows hoặc download |
| File Linux (ELF) | ❌ Không | EMBER không hỗ trợ ELF format |
| Script chạy trên Ubuntu | ✅ Có | Script Python chạy được trên mọi OS |
| Model load trên Ubuntu | ✅ Có | LightGBM model chạy được trên mọi OS |

**Kết luận**: Script có thể chạy trên Ubuntu, nhưng **chỉ test được file PE** (Windows executables). Muốn test file Linux, cần model khác được train cho ELF format.

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
