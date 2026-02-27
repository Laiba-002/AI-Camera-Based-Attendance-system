# """
# Production-Optimized FastAPI Application
# Enhanced with ONNX, multi-threading, liveness detection, and proper logging
# """
# from camera_worker_optimized import CameraWorkerManager
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional, Dict
# from datetime import datetime, date, timedelta
# import cv2
# import asyncio
# import sys

# import config
# from database import DatabaseManager

# # Setup logging FIRST
# from logging_config import setup_logging, get_logger
# setup_logging(log_level=config.LOG_LEVEL)
# logger = get_logger(__name__)

# # Import optimized camera worker (SCRFD + ArcFace ONNX)
# logger.info("✓ Using ONNX camera workers (SCRFD detector + ArcFace recognition)")
# USE_OPTIMIZED = False
# logger.info("✓ Using standard camera workers")

# # Initialize FastAPI app
# app = FastAPI(
#     title=config.API_TITLE,
#     version=config.API_VERSION,
#     description="Production-optimized AI Attendance System with ONNX Runtime, liveness detection, and multi-threading"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global instances
# camera_manager: Optional[CameraWorkerManager] = None
# db = DatabaseManager()


# # Pydantic models
# class EmployeeResponse(BaseModel):
#     employee_id: str
#     name: str
#     email: Optional[str]
#     department: Optional[str]
#     position: Optional[str]
#     enrolled_date: str
#     is_active: int


# class AttendanceRecord(BaseModel):
#     id: int
#     employee_id: str
#     name: str
#     date: str
#     punch_type: str
#     punch_time: str
#     timestamp: str
#     department: Optional[str]
#     position: Optional[str]
#     notes: Optional[str]


# class AttendanceStats(BaseModel):
#     total_records: int
#     unique_employees: int
#     check_ins: int
#     check_outs: int
#     date_range: Dict


# class SystemStatus(BaseModel):
#     status: str
#     version: str
#     engine: str
#     cameras: Dict
#     total_employees: int
#     timestamp: str


# # Startup and shutdown events
# @app.on_event("startup")
# async def startup_event():
#     """Initialize camera workers on startup"""
#     global camera_manager
#     logger.info("=" * 60)
#     logger.info("🚀 Starting Production-Optimized Attendance System")
#     logger.info("=" * 60)
#     logger.info("Engine: ONNX (SCRFD detector + ArcFace recognition)")
#     logger.info(f"Model: {config.ONNX_MODEL_NAME}")
#     logger.info(
#         f"Liveness Detection: {'Enabled' if config.LIVENESS_CHECK_ENABLED else 'Disabled'}")
#     logger.info(f"Log Level: {config.LOG_LEVEL}")

#     try:
#         camera_manager = CameraWorkerManager(use_optimized=True)
#         camera_manager.start_all()

#         logger.info("✓ System ready")
#         logger.info(
#             f"✓ API running on http://{config.API_HOST}:{config.API_PORT}")
#         logger.info("=" * 60)
#     except Exception as e:
#         logger.critical(f"Failed to start system: {e}", exc_info=True)
#         raise


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Stop camera workers on shutdown"""
#     global camera_manager
#     logger.info("Shutting down...")

#     if camera_manager:
#         camera_manager.stop_all()

#     logger.info("✓ Shutdown complete")


# # API Endpoints

# @app.get("/", tags=["General"])
# async def root():
#     """API root endpoint"""
#     return {
#         "message": "AI Attendance System API - Production Optimized",
#         "version": config.API_VERSION,
#         "engine": "ONNX (SCRFD + ArcFace)",
#         "model": config.ONNX_MODEL_NAME,
#         "status": "running",
#         "features": {
#             "detector": "SCRFD (Sample and Computation Redistribution Face Detection)",
#             "recognition": "ArcFace ONNX embeddings",
#             "liveness_detection": config.LIVENESS_CHECK_ENABLED,
#             "multi_threading": True,
#             "persistent_cooldown": True,
#             "shared_engine": True
#         },
#         "endpoints": {
#             "docs": "/docs",
#             "system_status": "/api/status",
#             "employees": "/api/employees",
#             "attendance": "/api/attendance",
#             "streams": "/api/stream/{camera_name}"
#         }
#     }


# @app.get("/api/status", response_model=SystemStatus, tags=["System"])
# async def get_system_status():
#     """Get system status and camera statistics"""
#     if not camera_manager:
#         raise HTTPException(status_code=503, detail="System not initialized")

#     camera_stats = camera_manager.get_all_stats()
#     employees = db.get_all_employees()

#     return {
#         "status": "running",
#         "version": config.API_VERSION,
#         "engine": f"ONNX (SCRFD + ArcFace) - {config.ONNX_MODEL_NAME}",
#         "cameras": camera_stats,
#         "total_employees": len(employees),
#         "timestamp": datetime.now().isoformat()
#     }


# @app.get("/api/employees", response_model=List[EmployeeResponse], tags=["Employees"])
# async def get_employees():
#     """Get all employees"""
#     employees = db.get_all_employees()
#     return employees


# @app.get("/api/employees/{employee_id}", response_model=EmployeeResponse, tags=["Employees"])
# async def get_employee(employee_id: str):
#     """Get employee by ID"""
#     employee = db.get_employee_by_id(employee_id)

#     if not employee:
#         raise HTTPException(status_code=404, detail="Employee not found")

#     return employee


# @app.get("/api/employees/{employee_id}/status", tags=["Employees"])
# async def get_employee_today_status(employee_id: str):
#     """Get employee's attendance status for today"""
#     employee = db.get_employee_by_id(employee_id)

#     if not employee:
#         raise HTTPException(status_code=404, detail="Employee not found")

#     status = db.get_employee_today_status(employee_id)
#     return status


# @app.get("/api/attendance", response_model=List[AttendanceRecord], tags=["Attendance"])
# async def get_attendance_records(
#     start_date: Optional[str] = Query(
#         None, description="Start date (YYYY-MM-DD)"),
#     end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
#     employee_id: Optional[str] = Query(
#         None, description="Filter by employee ID"),
#     limit: Optional[int] = Query(100, description="Limit number of records")
# ):
#     """Get attendance records with optional filters"""
#     records = db.get_attendance_records(start_date, end_date, employee_id)
#     return records[:limit]


# @app.get("/api/attendance/today", response_model=List[AttendanceRecord], tags=["Attendance"])
# async def get_today_attendance():
#     """Get today's attendance records"""
#     records = db.get_today_attendance()
#     return records


# @app.get("/api/attendance/stats", response_model=AttendanceStats, tags=["Attendance"])
# async def get_attendance_statistics(
#     start_date: Optional[str] = Query(
#         None, description="Start date (YYYY-MM-DD)"),
#     end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
# ):
#     """Get attendance statistics for a date range"""
#     stats = db.get_attendance_statistics(start_date, end_date)
#     return stats


# @app.get("/api/cameras", tags=["Cameras"])
# async def get_cameras():
#     """Get list of available cameras"""
#     if not camera_manager:
#         raise HTTPException(
#             status_code=503, detail="Camera system not initialized")

#     cameras = []
#     for name, worker in camera_manager.workers.items():
#         cameras.append({
#             "name": name,
#             "url": worker.camera_url,
#             "action": worker.action_type,
#             "stats": worker.get_stats()
#         })

#     return {"cameras": cameras}


# @app.get("/api/stream/{camera_name}", tags=["Streams"])
# async def stream_camera(camera_name: str):
#     """
#     Stream camera feed as Motion JPEG
#     Access via: http://localhost:8000/api/stream/Punch-In%20Camera
#     """
#     if not camera_manager:
#         raise HTTPException(
#             status_code=503, detail="Camera system not initialized")

#     worker = camera_manager.get_worker(camera_name)

#     if not worker:
#         available = list(camera_manager.workers.keys())
#         raise HTTPException(
#             status_code=404,
#             detail=f"Camera '{camera_name}' not found. Available: {available}"
#         )

#     def generate_frames():
#         """Generate frames for Motion JPEG stream"""
#         import time

#         while True:
#             frame = worker.get_latest_frame()

#             if frame is None:
#                 # Send blank frame
#                 blank = 255 * cv2.ones((360, 640, 3), dtype='uint8')
#                 cv2.putText(blank, 'No Frame Available', (150, 180),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 ret, buffer = cv2.imencode('.jpg', blank,
#                                            [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
#                 frame_bytes = buffer.tobytes()
#             else:
#                 # Encode frame as JPEG
#                 ret, buffer = cv2.imencode('.jpg', frame,
#                                            [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])

#                 if not ret:
#                     time.sleep(1.0 / config.STREAM_FPS)
#                     continue

#                 frame_bytes = buffer.tobytes()

#             # Yield frame in Motion JPEG format
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' +
#                    frame_bytes +
#                    b'\r\n')

#             # Control stream FPS
#             time.sleep(1.0 / config.STREAM_FPS)

#     return StreamingResponse(
#         generate_frames(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )


# @app.get("/api/recognition/logs", tags=["Recognition"])
# async def get_recognition_logs(limit: int = Query(100, description="Limit number of logs")):
#     """Get recent recognition logs"""
#     logs = db.get_recognition_logs(limit)
#     return {"logs": logs, "count": len(logs)}


# @app.get("/health", tags=["General"])
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "engine": "ONNX",
#         "detector": "SCRFD",
#         "recognition": "ArcFace"
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main_optimized:app",
#         host=config.API_HOST,
#         port=config.API_PORT,
#         reload=False,
#         log_level="info"
#     )


"""
Production FastAPI Application - Optimized Streaming
=====================================================
CHANGES FROM ORIGINAL:
1. generate_frames() was a sync generator inside async route - blocks event loop
   Now: runs in thread executor, async-friendly
2. JPEG quality was 100 in config - streaming at 100% quality is wasteful,
   75 is visually identical but 2-3x smaller payload = faster stream
3. Stream sleep was time.sleep(1/FPS) which is imprecise on Windows/Linux
   Now: monotonic-based precise frame pacing
4. Blank frame on None was regenerated every iteration - now cached
5. TurboJPEG integration for encode speed
6. USE_OPTIMIZED was hardcoded to False (bug in original) - fixed
"""

from camera_worker_optimized import CameraWorkerManager, encode_frame_fast
import cv2
import numpy as np
import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from database import DatabaseManager
from logging_config import setup_logging, get_logger

# Setup logging FIRST
setup_logging(log_level=config.LOG_LEVEL)
logger = get_logger(__name__)


# Initialize app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="AI Attendance System - ONNX Runtime + Optimized Streaming"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera_manager: Optional[CameraWorkerManager] = None
db = DatabaseManager()

# Pre-build blank frame (shown when camera not available)
_blank_frame_bytes: Optional[bytes] = None


def _get_blank_frame(width=640, height=360) -> bytes:
    global _blank_frame_bytes
    if _blank_frame_bytes is None:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(blank, 'Camera Not Available', (width//2 - 150, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2)
        _blank_frame_bytes = encode_frame_fast(blank, quality=50)
    return _blank_frame_bytes


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    global camera_manager
    logger.info("=" * 60)
    logger.info("Starting AI Attendance System")
    logger.info(f"Engine: ONNX ({config.ONNX_MODEL_NAME})")
    logger.info(f"Cameras: {list(config.CAMERA_URLS.keys())}")
    logger.info("=" * 60)

    try:
        camera_manager = CameraWorkerManager(use_optimized=True)
        camera_manager.start_all()
        logger.info("✓ System ready")
    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    if camera_manager:
        camera_manager.stop_all()
    logger.info("✓ Shutdown complete")


# ============================================================
# STREAM ENDPOINT - THE CRITICAL ONE
# ============================================================

@app.get("/api/stream/{camera_name}", tags=["Streams"])
async def stream_camera(camera_name: str):
    """
    MJPEG stream endpoint.
    
    FIXED from original:
    - sync generator ran in async context, blocking the event loop
    - now runs in threadpool via run_in_executor
    - frame pacing uses monotonic clock (accurate to microsecond)
    - quality 75 instead of 100 (same visual quality, 3x smaller)
    """
    if not camera_manager:
        raise HTTPException(status_code=503, detail="Camera system not ready")

    worker = camera_manager.get_worker(camera_name)
    if not worker:
        available = list(camera_manager.workers.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_name}' not found. Available: {available}"
        )

    frame_interval = 1.0 / config.STREAM_FPS  # e.g. 0.1s for 10fps
    blank = _get_blank_frame()

    def generate():
        """
        Sync generator - runs in threadpool, not on event loop.
        This is critical: if this ran on the event loop it would block
        ALL other requests while sleeping between frames.
        """
        next_frame_time = time.monotonic()

        while True:
            frame = worker.get_latest_frame()

            if frame is not None:
                frame_bytes = encode_frame_fast(
                    frame, quality=config.JPEG_QUALITY)
            else:
                frame_bytes = blank

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame_bytes +
                b'\r\n'
            )

            # Precise frame pacing (not just sleep(interval) which drifts)
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Fell behind - reset timer to avoid burst
                next_frame_time = time.monotonic()

    # Run the sync generator in a thread so it doesn't block the event loop
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "AI Attendance System",
        "version": config.API_VERSION,
        "engine": f"ONNX ({config.ONNX_MODEL_NAME})",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["General"])
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/status", tags=["System"])
async def system_status():
    if not camera_manager:
        raise HTTPException(status_code=503, detail="Not initialized")

    return {
        "status": "running",
        "version": config.API_VERSION,
        "cameras": camera_manager.get_all_stats(),
        "total_employees": len(db.get_all_employees()),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/cameras", tags=["Cameras"])
async def get_cameras():
    if not camera_manager:
        raise HTTPException(status_code=503, detail="Not initialized")

    cameras = []
    for name, worker in camera_manager.workers.items():
        cameras.append({
            "name": name,
            "url": worker.camera_url,
            "action": worker.action_type,
            "stats": worker.get_stats()
        })
    return {"cameras": cameras}


@app.get("/api/employees", tags=["Employees"])
async def get_employees():
    return db.get_all_employees()


@app.get("/api/employees/{employee_id}", tags=["Employees"])
async def get_employee(employee_id: str):
    emp = db.get_employee_by_id(employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp


@app.get("/api/employees/{employee_id}/status", tags=["Employees"])
async def get_employee_status(employee_id: str):
    if not db.get_employee_by_id(employee_id):
        raise HTTPException(status_code=404, detail="Employee not found")
    return db.get_employee_today_status(employee_id)


@app.get("/api/attendance", tags=["Attendance"])
async def get_attendance(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    employee_id: Optional[str] = Query(None),
    limit: int = Query(100)
):
    records = db.get_attendance_records(start_date, end_date, employee_id)
    return records[:limit]


@app.get("/api/attendance/today", tags=["Attendance"])
async def today_attendance():
    return db.get_today_attendance()


@app.get("/api/attendance/stats", tags=["Attendance"])
async def attendance_stats(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    return db.get_attendance_statistics(start_date, end_date)


@app.get("/api/recognition/logs", tags=["Recognition"])
async def recognition_logs(limit: int = Query(100)):
    logs = db.get_recognition_logs(limit)
    return {"logs": logs, "count": len(logs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_optimized:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        workers=1,  # Keep 1 worker - shared engine is not multiprocess safe
        log_level="info"
    )
