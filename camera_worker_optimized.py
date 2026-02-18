"""
Production-Optimized Camera Worker with Multi-Threading
- Separate threads for: frame reading, detection, recognition, database
- Queue-based producer-consumer pattern
- No blocking operations
- Handles high FPS without dropping frames
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

# Import production ONNX engine (SCRFD + ArcFace)
from face_recognition_engine_onnx import FaceRecognitionEngineONNX
from liveness_detection import LivenessDetector, SimpleLivenessDetector
from database import DatabaseManager

logger = logging.getLogger(__name__)


class CameraWorkerOptimized:
    """
    Production-optimized camera worker with multi-threading
    Separate threads for each pipeline stage
    """

    def __init__(self, camera_name: str, camera_url: str, action_type: str,
                 shared_face_engine=None, shared_liveness_detector=None):
        self.camera_name = camera_name
        self.camera_url = camera_url
        self.action_type = action_type

        self.db = DatabaseManager()

        # Use shared face recognition engine or create new one
        if shared_face_engine is not None:
            logger.info(f"[{camera_name}] Using shared recognition engine")
            self.face_engine = shared_face_engine
        else:
            # Create new ONNX engine (SCRFD + ArcFace)
            logger.info(
                f"[{camera_name}] Creating ONNX engine (SCRFD + ArcFace)")
            self.face_engine = FaceRecognitionEngineONNX()

        # Use shared liveness detector or create new one
        if shared_liveness_detector is not None:
            self.liveness_detector = shared_liveness_detector
        elif config.LIVENESS_CHECK_ENABLED:
            try:
                self.liveness_detector = LivenessDetector()
            except:
                logger.warning("Using simple liveness detector")
                self.liveness_detector = SimpleLivenessDetector()
        else:
            self.liveness_detector = None

        self.capture = None
        self.running = False

        # Threads
        self.reader_thread = None
        self.processor_thread = None
        self.database_thread = None

        # Queues for pipeline
        self.frame_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.detection_queue = queue.Queue(maxsize=10)
        self.attendance_queue = queue.Queue(maxsize=20)

        # Latest frame for streaming
        self.latest_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'attendance_marked': 0,
            'liveness_passed': 0,
            'liveness_failed': 0,
            'last_recognition': None,
            'fps': 0,
            'processing_fps': 0,
            'status': 'Initializing',
            'queue_sizes': {}
        }
        self.stats_lock = threading.Lock()

        # Frame processing control
        self.frame_count = 0
        self.last_process_time = time.time()

        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

    def connect(self) -> bool:
        """Connect to camera stream with improved settings"""
        logger.info(f"[{self.camera_name}] Connecting to {self.camera_url}")

        try:
            if self.capture is not None:
                self.capture.release()
                time.sleep(0.5)

            # Force TCP transport for RTSP (prevents AU header errors from UDP packet loss)
            self.capture = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)

            # RTSP settings - TCP transport prevents packetization errors
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                             config.RTSP_TIMEOUT_MS)
            self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                             config.RTSP_TIMEOUT_MS)

            # Use TCP flags for reliable stream (prevents AU header errors)
            self.capture.set(cv2.CAP_PROP_FOURCC,
                             cv2.VideoWriter_fourcc(*'H264'))
            # Limit FPS to reduce packet load
            self.capture.set(cv2.CAP_PROP_FPS, 10)

            # Test read
            for _ in range(5):
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    logger.info(f"[{self.camera_name}] ✓ Connected")
                    self._update_stats({'status': 'Connected'})
                    return True

            logger.error(f"[{self.camera_name}] ✗ Failed to read frames")
            return False

        except Exception as e:
            logger.error(f"[{self.camera_name}] Connection error: {e}")
            return False

    def start(self):
        """Start all worker threads"""
        if self.running:
            logger.warning(f"[{self.camera_name}] Already running")
            return

        if not self.connect():
            logger.error(f"[{self.camera_name}] Failed to connect")
            return

        self.running = True

        # Start pipeline threads
        self.reader_thread = threading.Thread(
            target=self._frame_reader_loop, daemon=True, name=f"{self.camera_name}-reader")
        self.processor_thread = threading.Thread(
            target=self._frame_processor_loop, daemon=True, name=f"{self.camera_name}-processor")
        self.database_thread = threading.Thread(
            target=self._database_worker_loop, daemon=True, name=f"{self.camera_name}-database")

        self.reader_thread.start()
        self.processor_thread.start()
        self.database_thread.start()

        logger.info(f"[{self.camera_name}] ✓ All threads started")
        self._update_stats({'status': 'Running'})

    def stop(self):
        """Stop all worker threads"""
        logger.info(f"[{self.camera_name}] Stopping...")
        self.running = False

        # Wait for threads to finish
        threads = [self.reader_thread,
                   self.processor_thread, self.database_thread]
        for thread in threads:
            if thread:
                thread.join(timeout=3)

        if self.capture:
            self.capture.release()

        self._update_stats({'status': 'Stopped'})
        logger.info(f"[{self.camera_name}] ✓ Stopped")

    def _frame_reader_loop(self):
        """Thread 1: Read frames from camera"""
        logger.info(f"[{self.camera_name}] Frame reader thread started")

        fps_start = time.time()
        fps_count = 0
        reconnect_attempts = 0
        max_reconnect_attempts = 5

        while self.running:
            try:
                ret, frame = self.capture.read()

                if not ret or frame is None:
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        logger.warning(f"[{self.camera_name}] Reconnecting...")
                        self._update_stats({'status': 'Reconnecting'})
                        if self.connect():
                            reconnect_attempts = 0
                        else:
                            time.sleep(3)
                    time.sleep(0.1)
                    continue

                reconnect_attempts = 0

                # Update stats
                with self.stats_lock:
                    self.stats['total_frames'] += 1

                fps_count += 1

                # Calculate FPS
                if fps_count >= 30:
                    elapsed = time.time() - fps_start
                    fps = fps_count / elapsed
                    self._update_stats({'fps': round(fps, 2)})
                    fps_start = time.time()
                    fps_count = 0

                # Store latest frame for streaming
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                # Put frame in processing queue (non-blocking)
                try:
                    self.frame_queue.put((time.time(), frame), block=False)
                except queue.Full:
                    # Queue full, skip this frame
                    pass

            except Exception as e:
                logger.error(f"[{self.camera_name}] Reader error: {e}")
                time.sleep(1)

        logger.info(f"[{self.camera_name}] Frame reader thread stopped")

    def _frame_processor_loop(self):
        """Thread 2: Process frames for face detection and recognition"""
        logger.info(f"[{self.camera_name}] Frame processor thread started")

        process_fps_start = time.time()
        process_fps_count = 0

        while self.running:
            try:
                # Get frame from queue (with timeout)
                try:
                    timestamp, frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Process every Nth frame for efficiency
                self.frame_count += 1
                if self.frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
                    continue

                # Resize for faster processing
                small_frame = cv2.resize(frame, (640, 360))

                # Detect and recognize faces
                logger.info(
                    f"[{self.camera_name}] Processing frame {self.frame_count}...")
                results = self.face_engine.process_frame(small_frame)

                with self.stats_lock:
                    self.stats['processed_frames'] += 1

                if results:
                    logger.info(
                        f"[{self.camera_name}] Found {len(results)} face(s)")
                    with self.stats_lock:
                        self.stats['faces_detected'] += len(results)

                    # Apply liveness check
                    if self.liveness_detector and config.LIVENESS_CHECK_ENABLED:
                        results = self._check_liveness(results, small_frame)

                    # Put recognized faces in attendance queue
                    for result in results:
                        if result['recognized']:
                            try:
                                self.attendance_queue.put({
                                    'employee_id': result['employee_id'],
                                    'name': result['name'],
                                    'confidence': result['confidence'],
                                    'timestamp': timestamp,
                                    'liveness_passed': result.get('liveness_passed', True)
                                }, block=False)
                            except queue.Full:
                                pass

                # Create annotated frame
                annotated = self._annotate_frame(frame, results)
                with self.frame_lock:
                    self.annotated_frame = annotated

                # Calculate processing FPS
                process_fps_count += 1
                if process_fps_count >= 10:
                    elapsed = time.time() - process_fps_start
                    process_fps = process_fps_count / elapsed
                    self._update_stats(
                        {'processing_fps': round(process_fps, 2)})
                    process_fps_start = time.time()
                    process_fps_count = 0

                # Update queue sizes
                self._update_stats({
                    'queue_sizes': {
                        'frames': self.frame_queue.qsize(),
                        'attendance': self.attendance_queue.qsize()
                    }
                })

            except Exception as e:
                logger.error(f"[{self.camera_name}] Processor error: {e}")
                time.sleep(0.5)

        logger.info(f"[{self.camera_name}] Frame processor thread stopped")

    def _database_worker_loop(self):
        """Thread 3: Handle database operations (attendance marking)"""
        logger.info(f"[{self.camera_name}] Database worker thread started")

        while self.running:
            try:
                # Get attendance record from queue
                try:
                    record = self.attendance_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                employee_id = record['employee_id']
                name = record['name']
                confidence = record['confidence']
                liveness_passed = record.get('liveness_passed', True)

                # Check cooldown using database (persistent)
                if not self.db.check_attendance_cooldown(employee_id, config.COOLDOWN_SECONDS):
                    continue

                # Mark attendance
                success, message = self.db.mark_attendance(
                    employee_id,
                    self.action_type,
                    confidence=confidence,
                    liveness_passed=liveness_passed
                )

                if success:
                    with self.stats_lock:
                        self.stats['faces_recognized'] += 1
                        self.stats['attendance_marked'] += 1
                        self.stats['last_recognition'] = {
                            'employee_id': employee_id,
                            'name': name,
                            'confidence': confidence,
                            'action': self.action_type,
                            'liveness': liveness_passed,
                            'timestamp': datetime.now().isoformat()
                        }

                    logger.info(
                        f"[{self.camera_name}] ✓✓✓ ATTENDANCE MARKED: {name} - {self.action_type} - {confidence:.2%} (ID: {employee_id})")
                else:
                    logger.warning(
                        f"[{self.camera_name}] ⚠ Cooldown active for {name} ({employee_id})")

            except Exception as e:
                logger.error(
                    f"[{self.camera_name}] Database worker error: {e}")
                time.sleep(0.5)

        logger.info(f"[{self.camera_name}] Database worker thread stopped")

    def _check_liveness(self, results: list, frame: np.ndarray) -> list:
        """Apply liveness detection to filter results"""
        filtered = []

        for result in results:
            employee_id = result.get('employee_id', 'unknown')
            bbox = result['bbox']
            x, y, w, h = bbox

            # Extract face region
            face_img = frame[y:y+h, x:x+w]

            # Check liveness
            if hasattr(self.liveness_detector, 'check_liveness'):
                is_live, score, reason = self.liveness_detector.check_liveness(
                    employee_id, face_img, bbox, frame
                )
            else:
                is_live, score, reason = self.liveness_detector.check_liveness_simple(
                    employee_id, face_img, bbox
                )

            if is_live:
                logger.info(
                    f"[{self.camera_name}]   ✓ Liveness: PASS (score: {score:.2f})")
            else:
                logger.warning(
                    f"[{self.camera_name}]   ✗ Liveness: FAIL - {reason}")

            # Add liveness info to result
            result['liveness_passed'] = is_live
            result['liveness_score'] = score
            result['liveness_reason'] = reason

            # Update stats
            with self.stats_lock:
                if is_live:
                    self.stats['liveness_passed'] += 1
                else:
                    self.stats['liveness_failed'] += 1

            # Only keep if liveness passed (or if not recognized)
            if is_live or not result['recognized']:
                filtered.append(result)
            else:
                logger.debug(f"Liveness failed for {employee_id}: {reason}")

        return filtered

    def _annotate_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated = frame.copy()

        for result in results:
            bbox = result['bbox']
            x, y, w, h = bbox

            # Scale bbox
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 360
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # Choose color based on recognition and liveness
            if result['recognized']:
                liveness_passed = result.get('liveness_passed', True)
                color = (0, 255, 0) if liveness_passed else (
                    0, 165, 255)  # Green or Orange
            else:
                color = (0, 255, 255)  # Yellow for unknown

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label
            if result['recognized']:
                label = f"{result['name']} ({result['confidence']:.0%})"
                if 'liveness_score' in result:
                    label += f" L:{result['liveness_score']:.1f}"
            else:
                label = "Unknown"

            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add camera info and stats
        info_text = f"{self.camera_name} | FPS:{self.stats.get('fps', 0)} | Proc:{self.stats.get('processing_fps', 0)}"
        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame for streaming"""
        with self.frame_lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_stats(self) -> Dict:
        """Get worker statistics"""
        with self.stats_lock:
            return self.stats.copy()

    def _update_stats(self, updates: Dict):
        """Update statistics"""
        with self.stats_lock:
            self.stats.update(updates)


class CameraWorkerManager:
    """Manages multiple optimized camera workers with shared recognition engine"""

    def __init__(self, use_optimized: bool = True):
        self.use_optimized = use_optimized
        self.workers = {}
        self.shared_face_engine = None
        self.shared_liveness_detector = None
        self.initialize_workers()

    def initialize_workers(self):
        """Initialize camera workers with shared recognition engine"""
        logger.info("Initializing camera workers...")

        if self.use_optimized:
            # Create ONE shared face recognition engine (SCRFD + ArcFace ONNX)
            logger.info(
                "Creating shared ONNX recognition engine (SCRFD detector + ArcFace embeddings)...")
            self.shared_face_engine = FaceRecognitionEngineONNX()

            # Create ONE shared liveness detector
            if config.LIVENESS_CHECK_ENABLED:
                try:
                    self.shared_liveness_detector = LivenessDetector()
                    logger.info("✓ Shared liveness detector created")
                except:
                    self.shared_liveness_detector = SimpleLivenessDetector()
                    logger.info("✓ Shared simple liveness detector created")

            logger.info(
                "✓ Shared ONNX engine ready - all cameras will use SCRFD + ArcFace")

        for camera_name, camera_url in config.CAMERA_URLS.items():
            action_type = config.CAMERA_ACTIONS.get(camera_name, 'check_in')

            worker = CameraWorkerOptimized(
                camera_name, camera_url, action_type,
                shared_face_engine=self.shared_face_engine,
                shared_liveness_detector=self.shared_liveness_detector
            )

            self.workers[camera_name] = worker

        logger.info(f"✓ Initialized {len(self.workers)} camera workers")

    def start_all(self):
        """Start all camera workers"""
        logger.info("Starting all camera workers...")
        for name, worker in self.workers.items():
            worker.start()
        logger.info("✓ All workers started")

    def stop_all(self):
        """Stop all camera workers"""
        logger.info("Stopping all camera workers...")
        for name, worker in self.workers.items():
            worker.stop()
        logger.info("✓ All workers stopped")

    def get_worker(self, camera_name: str) -> Optional[CameraWorkerOptimized]:
        """Get a specific worker"""
        return self.workers.get(camera_name)

    def get_all_stats(self) -> Dict:
        """Get statistics from all workers"""
        return {name: worker.get_stats() for name, worker in self.workers.items()}
