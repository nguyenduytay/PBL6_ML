@echo off
echo ========================================
echo    EMBER MALWARE DETECTION SETUP
echo ========================================
echo.

REM Kiểm tra Docker
echo [1/5] Kiểm tra Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker chưa được cài đặt!
    echo 📥 Download Docker Desktop từ: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)
echo ✅ Docker đã được cài đặt

REM Kiểm tra Docker daemon
echo [2/5] Kiểm tra Docker daemon...
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker daemon chưa chạy!
    echo 🔧 Hãy khởi động Docker Desktop
    pause
    exit /b 1
)
echo ✅ Docker daemon đang chạy

REM Build Docker image
echo [3/5] Build Docker image...
docker build -t ember-malware-detection .
if %errorlevel% neq 0 (
    echo ❌ Lỗi khi build Docker image!
    pause
    exit /b 1
)
echo ✅ Docker image đã được build

REM Tạo thư mục cần thiết
echo [4/5] Tạo thư mục cần thiết...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "samples" mkdir samples
echo ✅ Thư mục đã được tạo

REM Test EMBER
echo [5/5] Test EMBER...
docker run --rm ember-malware-detection python -c "import ember; print('EMBER OK!')"
if %errorlevel% neq 0 (
    echo ❌ Lỗi khi test EMBER!
    pause
    exit /b 1
)
echo ✅ EMBER hoạt động tốt

echo.
echo ========================================
echo    SETUP HOÀN TẤT!
echo ========================================
echo.
echo 🚀 Cách sử dụng:
echo 1. Chạy container: docker-compose up -d
echo 2. Vào container: docker exec -it ember-malware-detection /bin/bash
echo 3. Phân tích file: python ember_demo.py /workspace/samples/your_file.exe
echo.
echo 📚 Xem thêm: HUONG_DAN_CHAY_DU_AN.md
echo.
pause
