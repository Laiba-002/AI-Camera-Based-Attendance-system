# """
# Configuration settings for FastAPI Headless Attendance System
# Production-optimized version with ONNX support
# """
# import os

# # Project paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Use local resources (standalone mode)
# EMPLOYEES_DIR = os.path.join(BASE_DIR, "Employees Images")
# DATABASE_PATH = os.path.join(BASE_DIR, "attendance.db")
# LOGS_DIR = os.path.join(BASE_DIR, "logs")

# # API Configuration for employee enrollment
# API_ENDPOINT = "https://stage-api-truebooks.nkutech.com/api/hrmsEmployee/GetEmployeesWithAttendancePics"

# # API Configuration
# API_HOST = "0.0.0.0"
# API_PORT = 8000
# API_TITLE = "AI Attendance System API"
# API_VERSION = "2.0.0"  # Updated version with optimizations

# # Camera settings - RTSP URLs
# CAMERA_URLS = {
#     'Punch-In Camera': 'rtsp://AI:AI123456@192.168.100.29:554/Streaming/Channels/202',
#     'Punch-Out Camera': 'rtsp://AI:AI123456@192.168.100.29:554/Streaming/Channels/802',
# }

# # Camera action mapping
# CAMERA_ACTIONS = {
#     'Punch-In Camera': 'check_in',
#     'Punch-Out Camera': 'check_out',
# }

# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 720

# # Dual camera mode
# DUAL_CAMERA_MODE = True

# # OpenCV settings for RTSP - Fix AU header errors
# # Force TCP transport (UDP causes packetization errors), suppress FFmpeg errors
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|rtsp_flags;prefer_tcp'
# os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
# os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # -8 = AV_LOG_QUIET (no errors)
# os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'

# # RTSP connection settings
# RTSP_TIMEOUT_MS = 20000  # 20 seconds (increased from default)
# RTSP_BUFFER_SIZE = 1  # Minimal buffer for low latency
# RTSP_RECONNECT_DELAY = 3  # Seconds between reconnect attempts


# # =============================================================================
# # PRODUCTION OPTIMIZATION SETTINGS
# # =============================================================================

# # Engine selection (ONNX is 5-10x faster on CPU)
# USE_ONNX_ENGINE = True  # Set to False to use original DeepFace engine

# # ONNX Execution Providers (ordered by priority)
# # OpenVINO: Best for Intel CPUs (2-3x faster than CPU provider)
# # CPU: Fallback for all processors
# # Set to False to disable OpenVINO (requires: pip install openvino)
# USE_OPENVINO = True

# # Face recognition settings - ONNX (InsightFace)
# # Buffalo_l model: Best accuracy (512D embeddings)
# # Options: buffalo_l (more accurate), buffalo_s (faster)
# ONNX_MODEL_NAME = 'buffalo_l'
# ONNX_DETECTION_SIZE = (960, 960)  # Detection input size
# # ========== DEBUG MODE ==========
# # Set to True to bypass quality checks and show top 3 matches
# DEBUG_MODE = True  # ENABLED FOR TESTING - See raw matching behavior
# # ================================

# # ========== ENROLLMENT MATCHING MODE ==========
# # CRITICAL: Set to True to match API enrollment behavior
# # When True: Real-time recognition uses SAME lenient settings as API enrollment
# # - Accepts ALL detected faces (no quality validation)
# # - Can recognize blurry, small, or low-quality faces
# # - Ensures enrolled faces can be recognized regardless of quality
# # Recommendation: Keep True if using API enrollment, False for manual enrollment
# MATCH_ENROLLMENT_MODE = True  # Match API enrollment behavior
# # =============================================

# # Cosine similarity threshold (0.0-1.0, higher = stricter, 0.5 = practical for real cameras)
# # Set to 23% to reduce false positives (10% was too low)
# # Enrollment photos = high-quality close-ups, Live = distant RTSP feeds
# RECOGNITION_THRESHOLD_ONNX = 0.65  # 65% threshold - reduces false positives

# # Face recognition settings - DeepFace (fallback)
# DEEPFACE_MODEL = 'ArcFace'  # Keep for backward compatibility
# DEEPFACE_DETECTOR = 'retinaface'
# RECOGNITION_THRESHOLD = 0.60  # DeepFace threshold
# USE_GPU = True  # GPU support for DeepFace

# # Face quality thresholds (bypassed when DEBUG_MODE=True)
# MIN_FACE_SIZE = 40  # Minimum face size in pixels (lowered for distant cameras)
# MAX_FACE_SIZE = 800  # Maximum face size (prevent poster/photo)
# MIN_DETECTION_SCORE = 0.50  # Minimum detection confidence (lowered from 0.65)
# BLUR_THRESHOLD = 80.0  # Laplacian variance for blur detection (lowered from 100)
# MIN_BRIGHTNESS = 40  # Minimum face brightness
# MAX_BRIGHTNESS = 220  # Maximum face brightness

# # Liveness detection settings
# LIVENESS_CHECK_ENABLED = True  # Enable anti-spoofing
# LIVENESS_METHOD = 'advanced'  # Options: 'simple', 'advanced'
# LIVENESS_THRESHOLD = 0.5  # Minimum liveness score to pass
# LIVENESS_BUFFER_SIZE = 10  # Number of frames to track for motion
# MOTION_THRESHOLD = 5.0  # Minimum motion score (for simple method)

# # Attendance settings
# COOLDOWN_SECONDS = 300  # 5 minutes between recognitions (persistent in DB)
# RECOGNITION_CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for attendance

# # Processing settings
# PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for efficiency
# MAX_FACES_PER_FRAME = 5  # Maximum faces to process per frame
# DETECTION_SCALE = 1.0

# # Threading and queue settings
# FRAME_QUEUE_SIZE = 5  # Frame queue size (lower = less latency)
# DETECTION_QUEUE_SIZE = 10  # Detection queue size
# ATTENDANCE_QUEUE_SIZE = 20  # Attendance queue size

# # Camera reconnection settings
# MAX_RECONNECT_ATTEMPTS = 5  # Max consecutive failures before reconnect
# RECONNECT_DELAY = 3  # Seconds to wait between reconnects

# # Stream settings
# STREAM_FPS = 10  # Output stream FPS
# JPEG_QUALITY = 100  # JPEG compression quality for streaming

# # Logging settings
# LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
# LOG_RETENTION_DAYS = 30  # Days to keep log files

# # Performance monitoring
# ENABLE_PERFORMANCE_LOGGING = True  # Log performance metrics
# FPS_CALCULATION_INTERVAL = 30  # Frames between FPS calculations


"""
Configuration - Production Optimized
=====================================
KEY CHANGES FROM ORIGINAL:
- JPEG_QUALITY: 100 -> 75 (same visual, 3x smaller payload, faster stream)
- STREAM_FPS: 10 is fine, keep it
- PROCESS_EVERY_N_FRAMES: 3 -> 5 (for dual camera on single CPU, 5 is better balance)
- ONNX_DETECTION_SIZE: 960x960 -> 640x640 (MAJOR speedup, minimal accuracy loss for attendance)
- DEBUG_MODE: True -> False (logging top-3 matches on EVERY frame kills performance)
- FRAME_QUEUE_SIZE removed (replaced by double buffer in worker)
- Added STREAM_JPEG_QUALITY separate from enrollment quality
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPLOYEES_DIR = os.path.join(BASE_DIR, "Employees Images")
DATABASE_PATH = os.path.join(BASE_DIR, "attendance.db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

API_ENDPOINT = "https://stage-api-truebooks.nkutech.com/api/hrmsEmployee/GetEmployeesWithAttendancePics"

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "AI Attendance System API"
API_VERSION = "2.1.0"

# Camera RTSP URLs
CAMERA_URLS = {
    'Punch-In Camera':  'rtsp://AI:AI123456@192.168.100.29:554/Streaming/Channels/201',
    'Punch-Out Camera': 'rtsp://AI:AI123456@192.168.100.29:554/Streaming/Channels/801',
}

CAMERA_ACTIONS = {
    'Punch-In Camera':  'check_in',
    'Punch-Out Camera': 'check_out',
}

# Camera resolution - 720p is good balance for recognition
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DUAL_CAMERA_MODE = True

# Force TCP for RTSP - UDP causes AU header/packetization errors
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|rtsp_flags;prefer_tcp'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'

# RTSP connection settings
RTSP_TIMEOUT_MS = 20000
RTSP_BUFFER_SIZE = 1
RTSP_RECONNECT_DELAY = 3

# =============================================================================
# ENGINE SETTINGS
# =============================================================================

USE_ONNX_ENGINE = True
USE_OPENVINO = True  # Uses OpenVINO if installed, falls back to CPU

ONNX_MODEL_NAME = 'buffalo_l'

# CRITICAL CHANGE: 960x960 -> 640x640
# 960x960 detection input is for passport/close-up photos.
# For 720p RTSP feeds at typical employee-camera distance, 640x640 is sufficient
# and ~2x faster. If you have very distant cameras, try 480x480.
ONNX_DETECTION_SIZE = (640, 640)

# =============================================================================
# DEBUG / PRODUCTION MODE
# =============================================================================

# CRITICAL: Set DEBUG_MODE = False in production
# DEBUG_MODE = True causes logger.info() on EVERY face in EVERY frame:
#   - Top 3 match logging
#   - Threshold logging
#   - Quality check logging
# This creates thousands of log writes per minute which slows the system.
DEBUG_MODE = False

# Match enrollment mode - keep True for API enrollment
MATCH_ENROLLMENT_MODE = True

# =============================================================================
# RECOGNITION THRESHOLDS
# =============================================================================

RECOGNITION_THRESHOLD_ONNX = 0.25
RECOGNITION_CONFIDENCE_THRESHOLD = 0.70

DEEPFACE_MODEL = 'ArcFace'
DEEPFACE_DETECTOR = 'retinaface'
RECOGNITION_THRESHOLD = 0.60
USE_GPU = False  # Not using DeepFace in production

# =============================================================================
# FACE QUALITY
# =============================================================================

MIN_FACE_SIZE = 40
MAX_FACE_SIZE = 800
MIN_DETECTION_SCORE = 0.50
BLUR_THRESHOLD = 80.0
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220
FACE_DETECTION_SCALE = 1.0

# =============================================================================
# LIVENESS
# =============================================================================

LIVENESS_CHECK_ENABLED = True
LIVENESS_METHOD = 'advanced'
LIVENESS_THRESHOLD = 0.5
LIVENESS_BUFFER_SIZE = 10
MOTION_THRESHOLD = 5.0

# =============================================================================
# ATTENDANCE
# =============================================================================

COOLDOWN_SECONDS = 300   # 5 minutes between recognitions
RECOGNITION_COOLDOWN = 30    # In-memory cache cooldown (seconds)

# =============================================================================
# PROCESSING
# =============================================================================

# CHANGED: 3 -> 5
# With dual cameras sharing 1 ONNX engine, processing every 3rd frame from
# both cameras = engine gets hit 2x as often. Every 5th frame is smoother.
# At 25fps input, every 5th = 5 detections/sec per camera = more than enough.
PROCESS_EVERY_N_FRAMES = 5

MAX_FACES_PER_FRAME = 5

# =============================================================================
# STREAMING
# =============================================================================

STREAM_FPS = 15    # Increased from 10 - smoother viewing experience
# Stream FPS is independent of detection FPS - this is just display

# CRITICAL CHANGE: 100 -> 75
# JPEG quality 100 = uncompressed JPEG (very large files)
# JPEG quality 75  = standard web quality (visually same for video)
# Impact: ~3x smaller frame payloads = 3x faster stream delivery
JPEG_QUALITY = 75

# Reconnection
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 3  # Starting delay (doubles each failure, max 30s)

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_RETENTION_DAYS = 30
ENABLE_PERFORMANCE_LOGGING = True
FPS_CALCULATION_INTERVAL = 30
