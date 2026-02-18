@echo off
REM Production Deployment Script for AI Attendance System (Windows)
REM Run this script to deploy the application with Docker Compose

echo ==========================================
echo AI Attendance System - Production Deploy
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)
echo [OK] Docker found

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed.
    exit /b 1
)
echo [OK] Docker Compose found

echo.
echo Checking required files...

REM Check required files
set MISSING_FILES=0
for %%f in (main_optimized.py config.py camera_worker_optimized.py face_recognition_engine_onnx.py database.py vector_db.py liveness_detection.py logging_config.py requirements.txt Dockerfile docker-compose.yml) do (
    if exist "%%f" (
        echo [OK] %%f
    ) else (
        echo [ERROR] %%f ^(missing^)
        set MISSING_FILES=1
    )
)

if %MISSING_FILES%==1 (
    echo [ERROR] Missing required files. Cannot proceed.
    exit /b 1
)

echo.
echo Creating directories...
if not exist "logs" mkdir logs
if not exist "vector_db" mkdir vector_db
if not exist "Employees Images" mkdir "Employees Images"
echo [OK] Directories created

echo.
echo Checking for database and embeddings...

REM CRITICAL: Check if database exists
if not exist "attendance.db" (
    echo [ERROR] attendance.db NOT FOUND!
    echo.
    echo CRITICAL: Database must be created before Docker deployment
    echo.
    echo Run this command first:
    echo   python enroll_from_api.py --full
    echo.
    echo This will:
    echo   * Auto-create attendance.db
    echo   * Auto-create vector_db/
    echo   * Download employee photos
    echo   * Generate face embeddings
    echo.
    pause
    exit /b 1
)
echo [OK] attendance.db found

REM Check if vector database exists
if not exist "vector_db\embeddings.db" (
    echo [ERROR] vector_db\embeddings.db NOT FOUND!
    echo.
    echo CRITICAL: Embeddings must be created before Docker deployment
    echo.
    echo Run this command first:
    echo   python enroll_from_api.py --full
    echo.
    pause
    exit /b 1
)
echo [OK] vector_db\embeddings.db found

echo.
set /p rebuild="Do you want to rebuild the Docker image? (y/n): "
if /i "%rebuild%"=="y" (
    echo.
    echo Building Docker image...
    docker-compose build
    if errorlevel 1 (
        echo [ERROR] Failed to build Docker image
        exit /b 1
    )
    echo [OK] Docker image built successfully
)

echo.
echo Stopping existing containers...
docker-compose down 2>nul
echo [OK] Old containers stopped

echo.
echo Starting service...
docker-compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start service
    exit /b 1
)

echo.
echo Waiting for service to start...
timeout /t 5 /nobreak >nul

REM Check if service is healthy
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Service may not be fully started yet
    echo Check logs with: docker-compose logs -f
) else (
    echo [OK] Service is healthy!
)

echo.
echo ==========================================
echo Deployment Complete!
echo ==========================================
echo.
echo [OK] Service is running
echo.
echo Quick Links:
echo   * API Documentation: http://localhost:8000/docs
echo   * System Status:     http://localhost:8000/api/status
echo   * Health Check:      http://localhost:8000/health
echo.
echo Camera Streams:
echo   * Punch-In:  http://localhost:8000/api/stream/Punch-In%%20Camera
echo   * Punch-Out: http://localhost:8000/api/stream/Punch-Out%%20Camera
echo.
echo Useful Commands:
echo   * View logs:        docker-compose logs -f
echo   * Stop service:     docker-compose stop
echo   * Restart service:  docker-compose restart
echo   * Check status:     docker-compose ps
echo.

echo Current Status:
docker-compose ps

echo.
echo [OK] Deployment successful!
pause
