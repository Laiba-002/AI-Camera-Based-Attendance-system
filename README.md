# 🎯 AI Attendance System
## Production-Ready FastAPI Application with ONNX Face Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-optimized-orange.svg)](https://onnxruntime.ai/)

**Real-time face recognition attendance system using RTSP cameras, ONNX Runtime, and FastAPI.**

---

## 🎯 Overview

An enterprise-grade AI-powered attendance system that automatically recognizes employees from RTSP camera streams and records attendance with check-in/check-out tracking. Built with production-ready optimizations including **ONNX Runtime for 5-10x performance improvement** over traditional deep learning frameworks.

### **Key Capabilities**

- ✅ **Real-time Face Recognition** - SCRFD detection + ArcFace embeddings (512D)
- ✅ **Multi-Camera Support** - Dual camera system (Punch-In/Punch-Out)
- ✅ **RTSP Streaming** - Direct integration with IP cameras
- ✅ **Liveness Detection** - Anti-spoofing with photo/poster detection
- ✅ **REST API** - Complete FastAPI-based backend
- ✅ **Docker Ready** - Production-optimized containerization
- ✅ **Persistent Storage** - SQLite database with vector embeddings
- ✅ **Performance Optimized** - ONNX Runtime with OpenVINO acceleration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                   (main_optimized.py)                        │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─── Camera Worker Manager (Multi-threaded)
         │    ├─── Punch-In Camera  (RTSP Stream)
         │    └─── Punch-Out Camera (RTSP Stream)
         │
         ├─── Face Recognition Engine (ONNX)
         │    ├─── SCRFD Detector
         │    ├─── ArcFace Recognition
         │    └─── Vector Database (Embeddings)
         │
         ├─── Liveness Detection
         │    ├─── Quality Checks
         │    └─── Motion Analysis
         │
         └─── Database Manager
              ├─── SQLite (Employees, Attendance)
              └─── Cooldown Tracking
```

## ✨ Features

### **Face Recognition**
- **Detection**: SCRFD (Sample and Computation Redistribution Face Detection)
- **Recognition**: ArcFace ONNX model with 512-dimensional embeddings
- **Accuracy**: Configurable similarity threshold (default: 0.65)
- **Speed**: 5-10x faster than TensorFlow/DeepFace on CPU
- **Quality Checks**: Blur, brightness, size validation

### **Liveness Detection**
- Photo/poster detection
- Motion analysis
- Face quality validation
- Configurable sensitivity

### **Attendance Management**
- Automatic check-in/check-out
- Cooldown period (default: 5 minutes)
- Persistent cooldown across restarts
- Comprehensive attendance logs
- Real-time statistics

### **API Features**
- RESTful API with FastAPI
- Interactive API documentation (Swagger UI)
- Employee management endpoints
- Attendance records and statistics
- Live camera streaming (MJPEG)
- Health monitoring

## 📁 File Structure

### **Essential Files (12 Core Files)**

```
fastapi_headless/
├── main_optimized.py              # Main FastAPI application (PRODUCTION)
├── config.py                      # Configuration settings
├── camera_worker_optimized.py     # Camera processing (ONNX)
├── face_recognition_engine_onnx.py # ONNX recognition engine
├── database.py                    # Database manager
├── vector_db.py                   # Vector embeddings storage
├── liveness_detection.py          # Anti-spoofing module
├── logging_config.py              # Logging configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container build
├── docker-compose.yml             # Docker orchestration
└── .dockerignore                  # Docker build optimization
```

### **Support Files**

```
├── enroll_from_api.py             # Employee enrollment (CRITICAL)
├── check_embeddings.py            # Verify enrollment
├── check_db.py                    # Database verification
├── deploy.sh / deploy.bat         # Deployment scripts
└── *.md                           # Documentation
```

### **Data Directories (Auto-Created)**

```
├── attendance.db                  # SQLite database
├── vector_db/                     # Face embeddings
│   └── embeddings.db
├── Employees Images/              # Employee photos
└── logs/                          # Application logs
    └── attendance_system.log
```

**Full structure**: See [PRODUCTION_FILE_STRUCTURE.md](PRODUCTION_FILE_STRUCTURE.md)

---

## 🚀 Quick Start

### **Local Development**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure cameras (edit config.py)
# Update CAMERA_URLS with your RTSP streams

# 3. Enroll employees (auto-creates database)
python enroll_from_api.py --full

# 4. Run application
python main_optimized.py
# OR
uvicorn main_optimized:app --reload

# 5. Access API
# http://localhost:8000/docs
```

### **Production Deployment (Docker)**

⚠️ **CRITICAL**: Database files are NOT included. You must generate them first.

```bash
# 1. Update config.py (camera URLs)

# 2. Generate database & embeddings (REQUIRED FIRST)
python enroll_from_api.py --full
# This auto-creates:
#   - attendance.db
#   - vector_db/embeddings.db
#   - Employees Images/

# 3. Deploy with Docker
./deploy.sh          # Linux/Mac
# OR
deploy.bat           # Windows

# 4. Verify
curl http://localhost:8000/health
```

**Full deployment guide**: See [DEVOPS_INSTRUCTIONS.md](DEVOPS_INSTRUCTIONS.md)

---

## 🐳 Docker Deployment

### **Quick Deploy**

```bash
# Ensure database exists first
python enroll_from_api.py --full

# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### **Important Notes**

1. **Database must exist before Docker deployment**
2. Run `enroll_from_api.py --full` on host machine first
3. Docker mounts these volumes:
   - `./attendance.db` → Container database
   - `./vector_db/` → Container embeddings
   - `./Employees Images/` → Container photos
   - `./logs/` → Container logs

---

## 📡 API Documentation

### **Interactive Docs**

Once running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### **Key Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/status` | GET | System status & stats |
| `/api/employees` | GET | List all employees |
| `/api/employees/{id}` | GET | Get employee details |
| `/api/attendance` | GET | Attendance records |
| `/api/attendance/today` | GET | Today's attendance |
| `/api/cameras` | GET | Camera status |
| `/api/stream/{camera_name}` | GET | Live camera stream |

### **Example Requests**

```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/api/status

# Today's attendance
curl http://localhost:8000/api/attendance/today

# Camera streams (in browser)
http://localhost:8000/api/stream/Punch-In%20Camera
```

---

## ⚙️ Configuration

### **Key Settings in [config.py](config.py)**

```python
# RTSP Camera URLs (UPDATE THESE)
CAMERA_URLS = {
    'Punch-In Camera': 'rtsp://username:password@192.168.1.100:554/stream',
    'Punch-Out Camera': 'rtsp://username:password@192.168.1.101:554/stream',
}

# Recognition Settings
RECOGNITION_THRESHOLD_ONNX = 0.65  # 0.60-0.75 recommended
ONNX_MODEL_NAME = 'buffalo_l'      # buffalo_l (accurate) or buffalo_s (faster)

# Feature Flags
LIVENESS_CHECK_ENABLED = True      # Enable anti-spoofing
DEBUG_MODE = False                 # MUST be False in production

# Performance
PROCESS_EVERY_N_FRAMES = 3         # Process every Nth frame
COOLDOWN_SECONDS = 300             # 5 minutes between recognitions

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
```

---

## ⚡ Performance

### **Optimization Features**

- **ONNX Runtime**: 5-10x faster than TensorFlow
- **OpenVINO**: Intel CPU acceleration (2-3x boost)
- **Multi-threading**: Parallel camera processing
- **Frame Skipping**: Configurable (every 3rd frame)
- **Shared Engine**: Single recognition engine

### **Benchmarks**

| Configuration | FPS/Camera | CPU Usage | RAM Usage |
|---------------|------------|-----------|-----------|
| ONNX + OpenVINO | 8-12 | 50-80% | 1.5-2.5GB |
| ONNX (CPU only) | 5-8 | 70-90% | 1.2-2.0GB |

*4-core Intel CPU, 1280x720 streams*

---

## 🐛 Troubleshooting

### **Camera Not Connecting**

```bash
# Test RTSP URL
ffmpeg -rtsp_transport tcp -i "rtsp://url" -frames:v 1 test.jpg

# Check config.py camera URLs
# Verify network connectivity
```

### **Low Recognition Accuracy**

```python
# Adjust threshold in config.py
RECOGNITION_THRESHOLD_ONNX = 0.60  # Lower = more matches

# Enable debug mode
DEBUG_MODE = True  # See matching scores
```

### **No Employees Enrolled**

```bash
# Check enrollment
python check_embeddings.py

# Re-enroll
python enroll_from_api.py --full
```

### **Docker Issues**

```bash
# Check logs
docker-compose logs --tail=50

# Verify database exists
ls -la attendance.db vector_db/

# Common issue: Missing database
# Solution: Run enroll_from_api.py first
```

**More solutions**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

## 📚 Documentation

### **For Deployment**
- **[DEVOPS_INSTRUCTIONS.md](DEVOPS_INSTRUCTIONS.md)** - Complete deployment guide
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Detailed reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick commands

### **For Development**
- [PRODUCTION_FILE_STRUCTURE.md](PRODUCTION_FILE_STRUCTURE.md) - File organization
- [DEPLOYMENT_WORKFLOW.md](DEPLOYMENT_WORKFLOW.md) - Workflow guide
- [DEVOPS_PACKAGE_CHECKLIST.md](DEVOPS_PACKAGE_CHECKLIST.md) - Delivery checklist

---

## 🔧 Development

```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt

# Configure
nano config.py

# Run
python main_optimized.py
```

---

## 🔄 Updates & Maintenance

### **Update Application**

```bash
# Rebuild
docker-compose build --no-cache

# Restart
docker-compose down && docker-compose up -d
```

### **Re-enroll Employees**

```bash
docker-compose stop
python enroll_from_api.py --full
docker-compose start
```

### **Backup**

```bash
docker-compose stop
tar -czf backup_$(date +%Y%m%d).tar.gz attendance.db vector_db/
docker-compose start
```

---

## 📊 System Requirements

### **Minimum**
- CPU: 4 cores
- RAM: 4GB
- Storage: 10GB
- OS: Linux/Windows/macOS

### **Recommended**
- CPU: 8+ cores (Intel for OpenVINO)
- RAM: 8GB+
- Storage: 20GB+ SSD

---

## 📝 License

Copyright © 2026. All rights reserved.

---

## 🎉 Quick Start Summary

```bash
# 1. Configure
nano config.py

# 2. Enroll
python enroll_from_api.py --full

# 3. Deploy
./deploy.sh

# 4. Access
http://localhost:8000/docs
```

---

## 📞 Links

- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Status**: http://localhost:8000/api/status

---

**Version**: 2.0.0 (Production-Optimized with ONNX)  
**Status**: Production Ready ✅  
**Last Updated**: February 18, 2026

**Built with ❤️ using FastAPI, ONNX Runtime, and InsightFace**
