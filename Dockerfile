# Production Dockerfile for FastAPI AI Attendance System
# Optimized for ONNX Runtime with OpenVINO support
FROM python:3.10-slim

# Metadata
LABEL maintainer="Your Team"
LABEL description="AI Attendance System with ONNX Runtime"
LABEL version="2.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for OpenCV, ONNX, and RTSP
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # FFmpeg for RTSP streaming
    ffmpeg \
    libavcodec-extra \
    # Network utilities
    curl \
    wget \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Verify critical packages
    python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && \
    python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

# Copy application files (respects .dockerignore)
COPY . .

# Create necessary directories
RUN mkdir -p logs vector_db && \
    chmod -R 755 /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose FastAPI port
EXPOSE 8000

# Run the optimized application with uvicorn
CMD ["uvicorn", "main_optimized:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
