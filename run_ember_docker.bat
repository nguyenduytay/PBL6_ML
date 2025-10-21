@echo off
echo ========================================
echo    EMBER Malware Detection Docker
echo ========================================
echo.

REM Check if Docker is running
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Docker is running. Starting EMBER container...
echo.

REM Run EMBER container with interactive shell
docker run -it --rm -v "%cd%":/workspace ember-malware-detection /bin/bash

pause
