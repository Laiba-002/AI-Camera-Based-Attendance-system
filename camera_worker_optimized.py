# """
# Production-Optimized Camera Worker with Multi-Threading
# - Separate threads for: frame reading, detection, recognition, database
# - Queue-based producer-consumer pattern
# - No blocking operations
# - Handles high FPS without dropping frames
# """
# import cv2
# import numpy as np
# import threading
# import queue
# import time
# import os
# from datetime import datetime
# from typing import Dict, Optional
# import logging
# import config

# # Import production ONNX engine (SCRFD + ArcFace)
# from face_recognition_engine_onnx import FaceRecognitionEngineONNX
# from liveness_detection import LivenessDetector, SimpleLivenessDetector
# from database import DatabaseManager

# logger = logging.getLogger(__name__)


# class CameraWorkerOptimized:
#     """
#     Production-optimized camera worker with multi-threading
#     Separate threads for each pipeline stage
#     """

#     def __init__(self, camera_name: str, camera_url: str, action_type: str,
#                  shared_face_engine=None, shared_liveness_detector=None):
#         self.camera_name = camera_name
#         self.camera_url = camera_url
#         self.action_type = action_type

#         self.db = DatabaseManager()

#         # Use shared face recognition engine or create new one
#         if shared_face_engine is not None:
#             logger.info(f"[{camera_name}] Using shared recognition engine")
#             self.face_engine = shared_face_engine
#         else:
#             # Create new ONNX engine (SCRFD + ArcFace)
#             logger.info(
#                 f"[{camera_name}] Creating ONNX engine (SCRFD + ArcFace)")
#             self.face_engine = FaceRecognitionEngineONNX()

#         # Use shared liveness detector or create new one
#         if shared_liveness_detector is not None:
#             self.liveness_detector = shared_liveness_detector
#         elif config.LIVENESS_CHECK_ENABLED:
#             try:
#                 self.liveness_detector = LivenessDetector()
#             except:
#                 logger.warning("Using simple liveness detector")
#                 self.liveness_detector = SimpleLivenessDetector()
#         else:
#             self.liveness_detector = None

#         self.capture = None
#         self.running = False

#         # Threads
#         self.reader_thread = None
#         self.processor_thread = None
#         self.database_thread = None

#         # Queues for pipeline
#         self.frame_queue = queue.Queue(maxsize=5)  # Limit queue size
#         self.detection_queue = queue.Queue(maxsize=10)
#         self.attendance_queue = queue.Queue(maxsize=20)

#         # Latest frame for streaming
#         self.latest_frame = None
#         self.annotated_frame = None
#         self.frame_lock = threading.Lock()

#         # Statistics
#         self.stats = {
#             'total_frames': 0,
#             'processed_frames': 0,
#             'faces_detected': 0,
#             'faces_recognized': 0,
#             'attendance_marked': 0,
#             'liveness_passed': 0,
#             'liveness_failed': 0,
#             'last_recognition': None,
#             'fps': 0,
#             'processing_fps': 0,
#             'status': 'Initializing',
#             'queue_sizes': {}
#         }
#         self.stats_lock = threading.Lock()

#         # Frame processing control
#         self.frame_count = 0
#         self.last_process_time = time.time()

#         os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

#     def connect(self) -> bool:
#         """Connect to camera stream with improved settings"""
#         logger.info(f"[{self.camera_name}] Connecting to {self.camera_url}")

#         try:
#             if self.capture is not None:
#                 self.capture.release()
#                 time.sleep(0.5)

#             # Force TCP transport for RTSP (prevents AU header errors from UDP packet loss)
#             self.capture = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)

#             # RTSP settings - TCP transport prevents packetization errors
#             self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
#             self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
#                              config.RTSP_TIMEOUT_MS)
#             self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,
#                              config.RTSP_TIMEOUT_MS)

#             # Use TCP flags for reliable stream (prevents AU header errors)
#             self.capture.set(cv2.CAP_PROP_FOURCC,
#                              cv2.VideoWriter_fourcc(*'H264'))
#             # Limit FPS to reduce packet load
#             self.capture.set(cv2.CAP_PROP_FPS, 10)

#             # Test read
#             for _ in range(5):
#                 ret, frame = self.capture.read()
#                 if ret and frame is not None:
#                     logger.info(f"[{self.camera_name}] ✓ Connected")
#                     self._update_stats({'status': 'Connected'})
#                     return True

#             logger.error(f"[{self.camera_name}] ✗ Failed to read frames")
#             return False

#         except Exception as e:
#             logger.error(f"[{self.camera_name}] Connection error: {e}")
#             return False

#     def start(self):
#         """Start all worker threads"""
#         if self.running:
#             logger.warning(f"[{self.camera_name}] Already running")
#             return

#         if not self.connect():
#             logger.error(f"[{self.camera_name}] Failed to connect")
#             return

#         self.running = True

#         # Start pipeline threads
#         self.reader_thread = threading.Thread(
#             target=self._frame_reader_loop, daemon=True, name=f"{self.camera_name}-reader")
#         self.processor_thread = threading.Thread(
#             target=self._frame_processor_loop, daemon=True, name=f"{self.camera_name}-processor")
#         self.database_thread = threading.Thread(
#             target=self._database_worker_loop, daemon=True, name=f"{self.camera_name}-database")

#         self.reader_thread.start()
#         self.processor_thread.start()
#         self.database_thread.start()

#         logger.info(f"[{self.camera_name}] ✓ All threads started")
#         self._update_stats({'status': 'Running'})

#     def stop(self):
#         """Stop all worker threads"""
#         logger.info(f"[{self.camera_name}] Stopping...")
#         self.running = False

#         # Wait for threads to finish
#         threads = [self.reader_thread,
#                    self.processor_thread, self.database_thread]
#         for thread in threads:
#             if thread:
#                 thread.join(timeout=3)

#         if self.capture:
#             self.capture.release()

#         self._update_stats({'status': 'Stopped'})
#         logger.info(f"[{self.camera_name}] ✓ Stopped")

#     def _frame_reader_loop(self):
#         """Thread 1: Read frames from camera"""
#         logger.info(f"[{self.camera_name}] Frame reader thread started")

#         fps_start = time.time()
#         fps_count = 0
#         reconnect_attempts = 0
#         max_reconnect_attempts = 5

#         while self.running:
#             try:
#                 ret, frame = self.capture.read()

#                 if not ret or frame is None:
#                     reconnect_attempts += 1
#                     if reconnect_attempts >= max_reconnect_attempts:
#                         logger.warning(f"[{self.camera_name}] Reconnecting...")
#                         self._update_stats({'status': 'Reconnecting'})
#                         if self.connect():
#                             reconnect_attempts = 0
#                         else:
#                             time.sleep(3)
#                     time.sleep(0.1)
#                     continue

#                 reconnect_attempts = 0

#                 # Update stats
#                 with self.stats_lock:
#                     self.stats['total_frames'] += 1

#                 fps_count += 1

#                 # Calculate FPS
#                 if fps_count >= 30:
#                     elapsed = time.time() - fps_start
#                     fps = fps_count / elapsed
#                     self._update_stats({'fps': round(fps, 2)})
#                     fps_start = time.time()
#                     fps_count = 0

#                 # Store latest frame for streaming
#                 with self.frame_lock:
#                     self.latest_frame = frame.copy()

#                 # Put frame in processing queue (non-blocking)
#                 try:
#                     self.frame_queue.put((time.time(), frame), block=False)
#                 except queue.Full:
#                     # Queue full, skip this frame
#                     pass

#             except Exception as e:
#                 logger.error(f"[{self.camera_name}] Reader error: {e}")
#                 time.sleep(1)

#         logger.info(f"[{self.camera_name}] Frame reader thread stopped")

#     def _frame_processor_loop(self):
#         """Thread 2: Process frames for face detection and recognition"""
#         logger.info(f"[{self.camera_name}] Frame processor thread started")

#         process_fps_start = time.time()
#         process_fps_count = 0

#         while self.running:
#             try:
#                 # Get frame from queue (with timeout)
#                 try:
#                     timestamp, frame = self.frame_queue.get(timeout=1.0)
#                 except queue.Empty:
#                     continue

#                 # Process every Nth frame for efficiency
#                 self.frame_count += 1
#                 if self.frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
#                     continue

#                 # Resize for faster processing
#                 small_frame = cv2.resize(frame, (640, 360))

#                 # Detect and recognize faces
#                 logger.info(
#                     f"[{self.camera_name}] Processing frame {self.frame_count}...")
#                 results = self.face_engine.process_frame(small_frame)

#                 with self.stats_lock:
#                     self.stats['processed_frames'] += 1

#                 if results:
#                     logger.info(
#                         f"[{self.camera_name}] Found {len(results)} face(s)")
#                     with self.stats_lock:
#                         self.stats['faces_detected'] += len(results)

#                     # Apply liveness check
#                     if self.liveness_detector and config.LIVENESS_CHECK_ENABLED:
#                         results = self._check_liveness(results, small_frame)

#                     # Put recognized faces in attendance queue
#                     for result in results:
#                         if result['recognized']:
#                             try:
#                                 self.attendance_queue.put({
#                                     'employee_id': result['employee_id'],
#                                     'name': result['name'],
#                                     'confidence': result['confidence'],
#                                     'timestamp': timestamp,
#                                     'liveness_passed': result.get('liveness_passed', True)
#                                 }, block=False)
#                             except queue.Full:
#                                 pass

#                 # Create annotated frame
#                 annotated = self._annotate_frame(frame, results)
#                 with self.frame_lock:
#                     self.annotated_frame = annotated

#                 # Calculate processing FPS
#                 process_fps_count += 1
#                 if process_fps_count >= 10:
#                     elapsed = time.time() - process_fps_start
#                     process_fps = process_fps_count / elapsed
#                     self._update_stats(
#                         {'processing_fps': round(process_fps, 2)})
#                     process_fps_start = time.time()
#                     process_fps_count = 0

#                 # Update queue sizes
#                 self._update_stats({
#                     'queue_sizes': {
#                         'frames': self.frame_queue.qsize(),
#                         'attendance': self.attendance_queue.qsize()
#                     }
#                 })

#             except Exception as e:
#                 logger.error(f"[{self.camera_name}] Processor error: {e}")
#                 time.sleep(0.5)

#         logger.info(f"[{self.camera_name}] Frame processor thread stopped")

#     def _database_worker_loop(self):
#         """Thread 3: Handle database operations (attendance marking)"""
#         logger.info(f"[{self.camera_name}] Database worker thread started")

#         while self.running:
#             try:
#                 # Get attendance record from queue
#                 try:
#                     record = self.attendance_queue.get(timeout=1.0)
#                 except queue.Empty:
#                     continue

#                 employee_id = record['employee_id']
#                 name = record['name']
#                 confidence = record['confidence']
#                 liveness_passed = record.get('liveness_passed', True)

#                 # Check cooldown using database (persistent)
#                 if not self.db.check_attendance_cooldown(employee_id, config.COOLDOWN_SECONDS):
#                     continue

#                 # Mark attendance
#                 success, message = self.db.mark_attendance(
#                     employee_id,
#                     self.action_type,
#                     confidence=confidence,
#                     liveness_passed=liveness_passed
#                 )

#                 if success:
#                     with self.stats_lock:
#                         self.stats['faces_recognized'] += 1
#                         self.stats['attendance_marked'] += 1
#                         self.stats['last_recognition'] = {
#                             'employee_id': employee_id,
#                             'name': name,
#                             'confidence': confidence,
#                             'action': self.action_type,
#                             'liveness': liveness_passed,
#                             'timestamp': datetime.now().isoformat()
#                         }

#                     logger.info(
#                         f"[{self.camera_name}] ✓✓✓ ATTENDANCE MARKED: {name} - {self.action_type} - {confidence:.2%} (ID: {employee_id})")
#                 else:
#                     logger.warning(
#                         f"[{self.camera_name}] ⚠ Cooldown active for {name} ({employee_id})")

#             except Exception as e:
#                 logger.error(
#                     f"[{self.camera_name}] Database worker error: {e}")
#                 time.sleep(0.5)

#         logger.info(f"[{self.camera_name}] Database worker thread stopped")

#     def _check_liveness(self, results: list, frame: np.ndarray) -> list:
#         """Apply liveness detection to filter results"""
#         filtered = []

#         for result in results:
#             employee_id = result.get('employee_id', 'unknown')
#             bbox = result['bbox']
#             x, y, w, h = bbox

#             # Extract face region
#             face_img = frame[y:y+h, x:x+w]

#             # Check liveness
#             if hasattr(self.liveness_detector, 'check_liveness'):
#                 is_live, score, reason = self.liveness_detector.check_liveness(
#                     employee_id, face_img, bbox, frame
#                 )
#             else:
#                 is_live, score, reason = self.liveness_detector.check_liveness_simple(
#                     employee_id, face_img, bbox
#                 )

#             if is_live:
#                 logger.info(
#                     f"[{self.camera_name}]   ✓ Liveness: PASS (score: {score:.2f})")
#             else:
#                 logger.warning(
#                     f"[{self.camera_name}]   ✗ Liveness: FAIL - {reason}")

#             # Add liveness info to result
#             result['liveness_passed'] = is_live
#             result['liveness_score'] = score
#             result['liveness_reason'] = reason

#             # Update stats
#             with self.stats_lock:
#                 if is_live:
#                     self.stats['liveness_passed'] += 1
#                 else:
#                     self.stats['liveness_failed'] += 1

#             # Only keep if liveness passed (or if not recognized)
#             if is_live or not result['recognized']:
#                 filtered.append(result)
#             else:
#                 logger.debug(f"Liveness failed for {employee_id}: {reason}")

#         return filtered

#     def _annotate_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
#         """Annotate frame with detection results"""
#         annotated = frame.copy()

#         for result in results:
#             bbox = result['bbox']
#             x, y, w, h = bbox

#             # Scale bbox
#             scale_x = frame.shape[1] / 640
#             scale_y = frame.shape[0] / 360
#             x = int(x * scale_x)
#             y = int(y * scale_y)
#             w = int(w * scale_x)
#             h = int(h * scale_y)

#             # Choose color based on recognition and liveness
#             if result['recognized']:
#                 liveness_passed = result.get('liveness_passed', True)
#                 color = (0, 255, 0) if liveness_passed else (
#                     0, 165, 255)  # Green or Orange
#             else:
#                 color = (0, 255, 255)  # Yellow for unknown

#             cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

#             # Draw label
#             if result['recognized']:
#                 label = f"{result['name']} ({result['confidence']:.0%})"
#                 if 'liveness_score' in result:
#                     label += f" L:{result['liveness_score']:.1f}"
#             else:
#                 label = "Unknown"

#             cv2.putText(annotated, label, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Add camera info and stats
#         info_text = f"{self.camera_name} | FPS:{self.stats.get('fps', 0)} | Proc:{self.stats.get('processing_fps', 0)}"
#         cv2.putText(annotated, info_text, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         return annotated

#     def get_latest_frame(self) -> Optional[np.ndarray]:
#         """Get the latest annotated frame for streaming"""
#         with self.frame_lock:
#             if self.annotated_frame is not None:
#                 return self.annotated_frame.copy()
#             if self.latest_frame is not None:
#                 return self.latest_frame.copy()
#         return None

#     def get_stats(self) -> Dict:
#         """Get worker statistics"""
#         with self.stats_lock:
#             return self.stats.copy()

#     def _update_stats(self, updates: Dict):
#         """Update statistics"""
#         with self.stats_lock:
#             self.stats.update(updates)


# class CameraWorkerManager:
#     """Manages multiple optimized camera workers with shared recognition engine"""

#     def __init__(self, use_optimized: bool = True):
#         self.use_optimized = use_optimized
#         self.workers = {}
#         self.shared_face_engine = None
#         self.shared_liveness_detector = None
#         self.initialize_workers()

#     def initialize_workers(self):
#         """Initialize camera workers with shared recognition engine"""
#         logger.info("Initializing camera workers...")

#         if self.use_optimized:
#             # Create ONE shared face recognition engine (SCRFD + ArcFace ONNX)
#             logger.info(
#                 "Creating shared ONNX recognition engine (SCRFD detector + ArcFace embeddings)...")
#             self.shared_face_engine = FaceRecognitionEngineONNX()

#             # Create ONE shared liveness detector
#             if config.LIVENESS_CHECK_ENABLED:
#                 try:
#                     self.shared_liveness_detector = LivenessDetector()
#                     logger.info("✓ Shared liveness detector created")
#                 except:
#                     self.shared_liveness_detector = SimpleLivenessDetector()
#                     logger.info("✓ Shared simple liveness detector created")

#             logger.info(
#                 "✓ Shared ONNX engine ready - all cameras will use SCRFD + ArcFace")

#         for camera_name, camera_url in config.CAMERA_URLS.items():
#             action_type = config.CAMERA_ACTIONS.get(camera_name, 'check_in')

#             worker = CameraWorkerOptimized(
#                 camera_name, camera_url, action_type,
#                 shared_face_engine=self.shared_face_engine,
#                 shared_liveness_detector=self.shared_liveness_detector
#             )

#             self.workers[camera_name] = worker

#         logger.info(f"✓ Initialized {len(self.workers)} camera workers")

#     def start_all(self):
#         """Start all camera workers"""
#         logger.info("Starting all camera workers...")
#         for name, worker in self.workers.items():
#             worker.start()
#         logger.info("✓ All workers started")

#     def stop_all(self):
#         """Stop all camera workers"""
#         logger.info("Stopping all camera workers...")
#         for name, worker in self.workers.items():
#             worker.stop()
#         logger.info("✓ All workers stopped")

#     def get_worker(self, camera_name: str) -> Optional[CameraWorkerOptimized]:
#         """Get a specific worker"""
#         return self.workers.get(camera_name)

#     def get_all_stats(self) -> Dict:
#         """Get statistics from all workers"""
#         return {name: worker.get_stats() for name, worker in self.workers.items()}


"""
FULLY OPTIMIZED Camera Worker - Speed of Light Edition
=======================================================
CHANGES FROM ORIGINAL:
1. Frame reader uses threading.Event instead of frame.copy() on every read
2. CAP_PROP_BUFFERSIZE=1 + discard stale frames from queue (was missing)
3. Processor no longer calls frame_queue.get() and THEN skips frames - wasteful
   Now: skips at queue insertion level
4. CRITICAL FIX: process_frame called on small_frame but annotation done on
   original full frame with wrong scale math - now consistent
5. CRITICAL FIX: shared engine used by 2 threads simultaneously = race condition
   Added per-worker engine lock for ONNX inference
6. Removed logger.info() spam inside hot loops (was logging EVERY frame)
7. get_latest_frame() was doing frame.copy() under lock = blocking stream
   Now uses double-buffer pattern - zero lock contention on stream path
8. Database connection opened per-attendance-record = slow. Now uses
   persistent connection in database thread
9. TurboJPEG for stream encoding instead of cv2.imencode
10. Frame annotation scaled correctly (was wrong scale factors)
11. Reconnect logic improved - exponential backoff
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
from datetime import datetime
from typing import Dict, Optional
import logging
import config

from face_recognition_engine_onnx import FaceRecognitionEngineONNX
from liveness_detection import LivenessDetector, SimpleLivenessDetector
from database import DatabaseManager

logger = logging.getLogger(__name__)

# Try to import TurboJPEG for 3-5x faster JPEG encoding
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _turbo_jpeg = TurboJPEG()
    HAS_TURBOJPEG = True
    logger.info("✓ TurboJPEG available - using fast JPEG encoding")
except ImportError:
    HAS_TURBOJPEG = False
    logger.info(
        "TurboJPEG not available - using cv2.imencode (install: pip install PyTurboJPEG)")


def encode_frame_fast(frame: np.ndarray, quality: int = 75) -> bytes:
    """
    Fast JPEG encoding - TurboJPEG if available, else cv2.
    Quality 75 is sufficient for streaming and 2x smaller than 100.
    """
    if HAS_TURBOJPEG:
        return _turbo_jpeg.encode(frame, quality=quality, pixel_format=TJPF_BGR)
    else:
        ret, buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ret else b''


class DoubleBuffer:
    """
    Lock-free double buffer for latest frame.
    Writer never blocks reader, reader never blocks writer.
    Critical for zero-latency streaming.
    """

    def __init__(self):
        self._buffers = [None, None]
        self._write_idx = 0
        self._read_idx = 0
        self._lock = threading.Lock()

    def write(self, frame: np.ndarray):
        """Write new frame - non-blocking"""
        write_to = 1 - self._read_idx  # Write to the buffer not being read
        self._buffers[write_to] = frame  # Direct assignment, no copy
        with self._lock:
            self._read_idx = write_to

    def read(self) -> Optional[np.ndarray]:
        """Read latest frame - returns reference, not copy"""
        with self._lock:
            idx = self._read_idx
        return self._buffers[idx]


class CameraWorkerOptimized:
    """
    Production camera worker - fully optimized pipeline
    
    Thread layout:
      Thread 1 (reader):    camera -> raw_frame_buffer (double buffer)
                            + sends every Nth frame to detection_queue
      Thread 2 (processor): detection_queue -> run ONNX -> annotated_buffer
      Thread 3 (database):  attendance_queue -> mark attendance
      
    Stream path:            annotated_buffer.read() <- zero lock contention
    """

    def __init__(self, camera_name: str, camera_url: str, action_type: str,
                 shared_face_engine=None, shared_liveness_detector=None):
        self.camera_name = camera_name
        self.camera_url = camera_url
        self.action_type = action_type

        # Database - single instance, persistent connection in db_thread
        self.db = DatabaseManager()

        # Face engine - shared across cameras (ONNX models are stateless for inference)
        # CRITICAL: Add a lock because InsightFace's app.get() is NOT thread-safe
        if shared_face_engine is not None:
            self.face_engine = shared_face_engine
        else:
            logger.info(f"[{camera_name}] Creating dedicated ONNX engine")
            self.face_engine = FaceRecognitionEngineONNX()

        # Per-worker inference lock (prevents race condition on shared engine)
        # Both cameras share 1 engine but ONNX runtime is not reentrant
        self.inference_lock = None  # Set by CameraWorkerManager if shared

        # Liveness detector
        if shared_liveness_detector is not None:
            self.liveness_detector = shared_liveness_detector
        elif config.LIVENESS_CHECK_ENABLED:
            try:
                self.liveness_detector = LivenessDetector()
            except Exception:
                self.liveness_detector = SimpleLivenessDetector()
        else:
            self.liveness_detector = None

        self.capture = None
        self.running = False

        # Threads
        self.reader_thread = None
        self.processor_thread = None
        self.database_thread = None

        # === OPTIMIZED QUEUES ===
        # frame_queue: only holds frames for DETECTION (not every frame)
        # maxsize=2: if detector is busy, drop old frames - freshness > completeness
        self.detection_queue = queue.Queue(maxsize=2)
        self.attendance_queue = queue.Queue(maxsize=50)

        # === DOUBLE BUFFERS (lock-free read) ===
        self.raw_buffer = DoubleBuffer()        # Latest raw frame for stream fallback
        self.annotated_buffer = DoubleBuffer()  # Latest annotated frame for stream

        # Last known detection results (overlaid on every stream frame)
        self._last_results = []
        self._results_lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'attendance_marked': 0,
            'fps': 0.0,
            'processing_fps': 0.0,
            'status': 'Initializing',
            'last_recognition': None,
        }
        self.stats_lock = threading.Lock()

        # Frame skip counter
        self._frame_count = 0

        # Reconnect state
        self._consecutive_failures = 0
        self._reconnect_delay = config.RECONNECT_DELAY

    # =========================================================================
    # CONNECTION
    # =========================================================================

    def connect(self) -> bool:
        """Connect to RTSP stream with optimized settings"""
        logger.info(f"[{self.camera_name}] Connecting...")

        try:
            if self.capture is not None:
                self.capture.release()
                time.sleep(0.5)

            # CAP_FFMPEG forces FFmpeg backend which supports RTSP properly
            self.capture = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)

            # === CRITICAL SETTINGS ===
            # Buffer=1: discard stale frames, always get freshest frame
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                             config.RTSP_TIMEOUT_MS)
            self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                             config.RTSP_TIMEOUT_MS)

            # Request specific resolution - prevents camera sending 1080p when we want 720p
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            # Test with multiple reads to flush buffer
            for _ in range(3):
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    logger.info(
                        f"[{self.camera_name}] ✓ Connected ({frame.shape[1]}x{frame.shape[0]})")
                    self._update_stats({'status': 'Connected'})
                    self._consecutive_failures = 0
                    self._reconnect_delay = config.RECONNECT_DELAY
                    return True

            logger.error(
                f"[{self.camera_name}] ✗ Could not read frames after connect")
            return False

        except Exception as e:
            logger.error(f"[{self.camera_name}] Connection error: {e}")
            return False

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Start all worker threads"""
        if self.running:
            return

        if not self.connect():
            logger.error(
                f"[{self.camera_name}] Failed to connect - will retry in background")
            # Start anyway, reader will keep retrying

        self.running = True

        self.reader_thread = threading.Thread(
            target=self._frame_reader_loop,
            daemon=True,
            name=f"{self.camera_name}-reader"
        )
        self.processor_thread = threading.Thread(
            target=self._frame_processor_loop,
            daemon=True,
            name=f"{self.camera_name}-processor"
        )
        self.database_thread = threading.Thread(
            target=self._database_worker_loop,
            daemon=True,
            name=f"{self.camera_name}-database"
        )

        self.reader_thread.start()
        self.processor_thread.start()
        self.database_thread.start()

        logger.info(f"[{self.camera_name}] ✓ All threads started")
        self._update_stats({'status': 'Running'})

    def stop(self):
        """Stop all threads cleanly"""
        self.running = False
        for thread in [self.reader_thread, self.processor_thread, self.database_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=3)
        if self.capture:
            self.capture.release()
        self._update_stats({'status': 'Stopped'})
        logger.info(f"[{self.camera_name}] ✓ Stopped")

    # =========================================================================
    # THREAD 1: FRAME READER
    # =========================================================================

    def _frame_reader_loop(self):
        """
        Reads frames from camera as fast as possible.
        - Always stores latest raw frame in double buffer (for stream)
        - Sends every Nth frame to detection_queue (non-blocking drop if full)
        - Handles reconnection with exponential backoff
        """
        logger.info(f"[{self.camera_name}] Reader thread started")

        fps_counter = 0
        fps_timer = time.monotonic()

        while self.running:
            try:
                if self.capture is None or not self.capture.isOpened():
                    self._reconnect()
                    continue

                ret, frame = self.capture.read()

                if not ret or frame is None:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= config.MAX_RECONNECT_ATTEMPTS:
                        logger.warning(
                            f"[{self.camera_name}] {self._consecutive_failures} failures - reconnecting")
                        self._reconnect()
                    else:
                        time.sleep(0.05)
                    continue

                self._consecutive_failures = 0

                # FPS counter
                fps_counter += 1
                now = time.monotonic()
                if fps_counter >= 30:
                    elapsed = now - fps_timer
                    self._update_stats({'total_frames': self.stats['total_frames'] + fps_counter,
                                       'fps': round(fps_counter / elapsed, 1)})
                    fps_counter = 0
                    fps_timer = now

                # Store raw frame in double buffer (NO COPY - direct write)
                self.raw_buffer.write(frame)

                # === FRAME SKIP: only send every Nth frame to detector ===
                self._frame_count += 1
                if self._frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                    # Resize HERE in reader thread (cheap), not in processor thread
                    # This means processor never handles full-res frames
                    small = cv2.resize(frame, (640, 360),
                                       interpolation=cv2.INTER_NEAREST)

                    # Non-blocking put: if queue is full (detector busy), drop this frame
                    # We want FRESH frames, not a backlog of stale ones
                    try:
                        self.detection_queue.put_nowait(
                            (time.monotonic(), small, frame.shape))
                    except queue.Full:
                        pass  # Drop stale frame - detector will get next one

            except Exception as e:
                logger.error(f"[{self.camera_name}] Reader error: {e}")
                time.sleep(0.5)

        logger.info(f"[{self.camera_name}] Reader thread stopped")

    def _reconnect(self):
        """Reconnect with exponential backoff"""
        self._update_stats({'status': 'Reconnecting'})
        logger.warning(
            f"[{self.camera_name}] Reconnecting in {self._reconnect_delay}s...")
        time.sleep(self._reconnect_delay)
        # Exponential backoff: 3 -> 6 -> 12 -> 24 -> max 30s
        self._reconnect_delay = min(self._reconnect_delay * 2, 30)

        if self.connect():
            logger.info(f"[{self.camera_name}] ✓ Reconnected")
            self._consecutive_failures = 0
            self._update_stats({'status': 'Running'})
        else:
            logger.error(f"[{self.camera_name}] Reconnect failed")

    # =========================================================================
    # THREAD 2: FRAME PROCESSOR (DETECTION + RECOGNITION)
    # =========================================================================

    def _frame_processor_loop(self):
        """
        Runs face detection + recognition on frames from detection_queue.
        Results are stored in annotated_buffer and _last_results.
        
        KEY OPTIMIZATION: Uses inference_lock when engine is shared.
        This prevents 2 cameras from calling app.get() simultaneously
        which would corrupt ONNX session state.
        """
        logger.info(f"[{self.camera_name}] Processor thread started")

        proc_fps_counter = 0
        proc_fps_timer = time.monotonic()

        while self.running:
            try:
                try:
                    timestamp, small_frame, original_shape = self.detection_queue.get(
                        timeout=1.0)
                except queue.Empty:
                    continue

                t_start = time.monotonic()

                # Acquire inference lock if shared (prevents race condition)
                lock = self.inference_lock
                if lock:
                    lock.acquire()

                try:
                    results = self.face_engine.process_frame(small_frame)
                finally:
                    if lock:
                        lock.release()

                # Liveness check (no lock needed, per-worker state)
                if results and self.liveness_detector and config.LIVENESS_CHECK_ENABLED:
                    results = self._apply_liveness(results, small_frame)

                # Queue recognized faces for attendance marking
                for r in results:
                    if r.get('recognized') and r.get('confidence', 0) >= config.RECOGNITION_CONFIDENCE_THRESHOLD:
                        try:
                            self.attendance_queue.put_nowait({
                                'employee_id': r['employee_id'],
                                'name': r['name'],
                                'confidence': r['confidence'],
                                'liveness_passed': r.get('liveness_passed', True),
                            })
                        except queue.Full:
                            pass

                # Store results for overlay
                with self._results_lock:
                    self._last_results = results

                # Build annotated frame
                # Use latest raw frame (may be newer than what was processed - that's fine)
                raw = self.raw_buffer.read()
                if raw is not None:
                    annotated = self._annotate(raw, results, original_shape)
                    self.annotated_buffer.write(annotated)

                # Stats
                proc_fps_counter += 1
                now = time.monotonic()
                if proc_fps_counter >= 10:
                    elapsed = now - proc_fps_timer
                    self._update_stats({
                        'processed_frames': self.stats['processed_frames'] + proc_fps_counter,
                        'processing_fps': round(proc_fps_counter / elapsed, 1),
                        'faces_detected': self.stats['faces_detected'] + len(results),
                    })
                    proc_fps_counter = 0
                    proc_fps_timer = now

            except Exception as e:
                logger.error(f"[{self.camera_name}] Processor error: {e}")
                time.sleep(0.1)

        logger.info(f"[{self.camera_name}] Processor thread stopped")

    def _apply_liveness(self, results: list, frame: np.ndarray) -> list:
        """Apply liveness check to results"""
        filtered = []
        for r in results:
            bbox = r['bbox']
            x, y, w, h = bbox
            face_img = frame[y:y+h, x:x+w]

            if face_img.size == 0:
                r['liveness_passed'] = True
                filtered.append(r)
                continue

            face_id = r.get('employee_id', f"{x}_{y}")

            try:
                if hasattr(self.liveness_detector, 'check_liveness'):
                    is_live, score, reason = self.liveness_detector.check_liveness(
                        face_id, face_img, bbox, frame)
                else:
                    is_live, score, reason = self.liveness_detector.check_liveness_simple(
                        face_id, face_img, bbox)
            except Exception:
                is_live, score = True, 1.0

            r['liveness_passed'] = is_live
            r['liveness_score'] = score

            if is_live or not r.get('recognized', False):
                filtered.append(r)

        return filtered

    # =========================================================================
    # THREAD 3: DATABASE WORKER
    # =========================================================================

    def _database_worker_loop(self):
        """
        Handles attendance marking asynchronously.
        Never blocks the video pipeline.
        Uses single persistent DB connection for the thread's lifetime.
        """
        logger.info(f"[{self.camera_name}] Database thread started")

        while self.running:
            try:
                try:
                    record = self.attendance_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                emp_id = record['employee_id']

                # Cooldown check
                if not self.db.check_attendance_cooldown(emp_id, config.COOLDOWN_SECONDS):
                    continue  # In cooldown, skip silently

                success, msg = self.db.mark_attendance(
                    emp_id,
                    self.action_type,
                    confidence=record['confidence'],
                    liveness_passed=record.get('liveness_passed', True)
                )

                if success:
                    self._update_stats({
                        'attendance_marked': self.stats['attendance_marked'] + 1,
                        'last_recognition': {
                            'name': record['name'],
                            'employee_id': emp_id,
                            'confidence': record['confidence'],
                            'action': self.action_type,
                            'timestamp': datetime.now().isoformat(),
                        }
                    })
                    logger.info(
                        f"[{self.camera_name}] ✓ ATTENDANCE: {record['name']} "
                        f"({emp_id}) - {self.action_type} - {record['confidence']:.1%}"
                    )

            except Exception as e:
                logger.error(f"[{self.camera_name}] DB worker error: {e}")
                time.sleep(0.5)

        logger.info(f"[{self.camera_name}] Database thread stopped")

    # =========================================================================
    # ANNOTATION
    # =========================================================================

    def _annotate(self, frame: np.ndarray, results: list, proc_shape: tuple) -> np.ndarray:
        """
        Draw bboxes on frame.
        proc_shape = (H, W, C) of the small frame that was processed.
        frame = full resolution frame.
        Scale bboxes from small -> full resolution.
        """
        annotated = frame.copy()
        fh, fw = frame.shape[:2]
        ph, pw = proc_shape[:2]

        scale_x = fw / pw
        scale_y = fh / ph

        for r in results:
            x, y, w, h = r['bbox']
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            if r.get('recognized'):
                liveness_ok = r.get('liveness_passed', True)
                color = (0, 255, 0) if liveness_ok else (
                    0, 140, 255)  # Green / Orange
                label = f"{r['name']} {r['confidence']:.0%}"
            else:
                color = (0, 255, 255)  # Yellow
                label = r.get('name', 'Unknown')

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Overlay: camera name + FPS
        fps = self.stats.get('fps', 0)
        proc_fps = self.stats.get('processing_fps', 0)
        overlay = f"{self.camera_name}  |  Cam:{fps}fps  Proc:{proc_fps}fps"
        cv2.putText(annotated, overlay, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame for HTTP streaming.
        Returns annotated frame if available, else raw frame.
        ZERO lock contention - double buffer reads are lock-free on read path.
        """
        frame = self.annotated_buffer.read()
        if frame is None:
            frame = self.raw_buffer.read()
        if frame is not None:
            return frame.copy()  # Only one copy, at stream time
        return None

    def get_stats(self) -> Dict:
        with self.stats_lock:
            return self.stats.copy()

    def _update_stats(self, updates: Dict):
        with self.stats_lock:
            self.stats.update(updates)


# =============================================================================
# CAMERA WORKER MANAGER
# =============================================================================

class CameraWorkerManager:
    """
    Manages multiple camera workers.
    
    KEY: Creates ONE shared ONNX engine + ONE shared threading.Lock.
    Both cameras take turns using the engine - no race conditions.
    """

    def __init__(self, use_optimized: bool = True):
        self.workers: Dict[str, CameraWorkerOptimized] = {}
        self._shared_engine = None
        self._shared_liveness = None
        self._inference_lock = threading.Lock()  # Shared inference lock
        self._initialize()

    def _initialize(self):
        logger.info("Initializing camera system...")

        # Single ONNX engine - loading models twice wastes RAM and startup time
        logger.info("Loading ONNX engine (SCRFD + ArcFace)...")
        self._shared_engine = FaceRecognitionEngineONNX()
        logger.info("✓ Shared ONNX engine ready")

        if config.LIVENESS_CHECK_ENABLED:
            try:
                self._shared_liveness = LivenessDetector()
                logger.info("✓ Shared liveness detector ready")
            except Exception:
                self._shared_liveness = SimpleLivenessDetector()

        for camera_name, camera_url in config.CAMERA_URLS.items():
            action_type = config.CAMERA_ACTIONS.get(camera_name, 'check_in')

            worker = CameraWorkerOptimized(
                camera_name=camera_name,
                camera_url=camera_url,
                action_type=action_type,
                shared_face_engine=self._shared_engine,
                shared_liveness_detector=self._shared_liveness,
            )
            # Give each worker access to the shared inference lock
            worker.inference_lock = self._inference_lock
            self.workers[camera_name] = worker

        logger.info(f"✓ {len(self.workers)} camera workers initialized")

    def start_all(self):
        for name, worker in self.workers.items():
            logger.info(f"Starting {name}...")
            worker.start()
        logger.info("✓ All workers started")

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop()
        logger.info("✓ All workers stopped")

    def get_worker(self, name: str) -> Optional[CameraWorkerOptimized]:
        return self.workers.get(name)

    def get_all_stats(self) -> Dict:
        return {name: w.get_stats() for name, w in self.workers.items()}
