"""
Production-Optimized Face Recognition Engine using Pure ONNX Runtime
Architecture:
- Detection: SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
  * 10x faster than RetinaFace/MTCNN on CPU
  * Better accuracy at small face sizes
  * Optimized for real-time applications
  
- Recognition: ArcFace ONNX embeddings
  * State-of-the-art accuracy (96-98%)
  * 512D embeddings (optimal balance)
  * 20x faster than TensorFlow version
  
- Backend: ONNX Runtime with OpenVINO (Intel CPU optimization)
  * No TensorFlow/PyTorch dependencies
  * Pure C++ execution for speed
  * 2-3x faster with OpenVINO on Intel CPUs
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import logging
from insightface.app import FaceAnalysis
import config
from vector_db import VectorDBManager
from database import DatabaseManager

# Setup logging
logger = logging.getLogger(__name__)


class FaceRecognitionEngineONNX:
    """Production-optimized face recognition engine using ONNX Runtime"""

    def __init__(self):
        self.db = DatabaseManager()
        self.vector_db = VectorDBManager()
        self.known_face_encodings = []
        self.known_employee_ids = []
        self.known_employee_names = []
        self.recognition_cache = {}  # For cooldown management
        self.using_gpu = False

        logger.info("Initializing ONNX Face Recognition Engine...")

        # Initialize InsightFace with ONNX backend
        try:
            # Try with OpenVINO for Intel CPU acceleration, fallback to CPU
            providers = []

            # Check if OpenVINO is available and enabled (faster on Intel CPUs)
            if config.USE_OPENVINO:
                try:
                    import openvino
                    providers.append('OpenVINOExecutionProvider')
                    logger.info(
                        "✓ OpenVINO detected - Intel CPU acceleration enabled")
                except ImportError:
                    logger.info(
                        "OpenVINO not installed - install with: pip install openvino")

            # Always add CPU as fallback
            providers.append('CPUExecutionProvider')

            logger.info(f"Using execution providers: {providers}")

            # Only load detection and recognition models (skip landmark_3d_68, landmark_2d_106, genderage)
            self.app = FaceAnalysis(
                # Use model from config (buffalo_l or buffalo_s)
                name=config.ONNX_MODEL_NAME,
                allowed_modules=['detection', 'recognition'],
                providers=providers
            )
            # Lower detection threshold for enrollment (0.3 = very lenient)
            self.app.prepare(ctx_id=-1, det_size=config.ONNX_DETECTION_SIZE, det_thresh=0.45)
            logger.info(
                f"✓ InsightFace {config.ONNX_MODEL_NAME} loaded (detection + recognition only) with {providers[0]}")
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            raise

        # Quality thresholds (adjusted for RTSP cameras)
        # Minimum face size in pixels (lowered for distant faces)
        self.min_face_size = 20  # Lowered from 40 to accept smaller faces
        self.max_face_size = 800  # Maximum face size
        # Minimum detection confidence (50% - standard for enrollment)
        self.min_detection_score = 0.5  # Restored to 0.5 from 0.3
        # Laplacian variance threshold (lowered for RTSP compression)
        self.blur_threshold = 30.0  # Lowered from 50.0 to accept blurrier images

        # Load employee encodings
        self.load_employee_encodings()
        logger.info(
            f"✓ Recognition engine ready ({len(self.known_employee_ids)} employees)")

    def load_employee_encodings(self):
        """Load employee face encodings from vector database"""
        logger.info("Loading employee face encodings...")

        try:
            results = self.vector_db.get_all_encodings()

            if not results or len(results) == 0:
                logger.warning(
                    "⚠ No employee encodings found in vector database")
                return

            self.known_employee_ids = []
            self.known_face_encodings = []
            self.known_employee_names = []

            for emp_id, embedding, metadata in results:
                # Embeddings should already be normalized from enrollment
                # But normalize again as safety measure to ensure consistency
                embedding_norm = np.linalg.norm(embedding)
                if embedding_norm > 0:
                    embedding_normalized = embedding / embedding_norm
                else:
                    # Skip invalid embeddings
                    logger.warning(f"Skipping invalid embedding for {emp_id} (zero norm)")
                    continue
                    
                self.known_employee_ids.append(emp_id)
                self.known_face_encodings.append(embedding_normalized)
                self.known_employee_names.append(
                    metadata.get('name', 'Unknown'))

            logger.info(
                f"✓ Loaded {len(self.known_employee_ids)} employee encodings")

        except Exception as e:
            logger.error(f"Error loading encodings: {str(e)}")
            self.known_employee_ids = []
            self.known_face_encodings = []
            self.known_employee_names = []

    def check_face_quality(self, face_img: np.ndarray, bbox: List[int], enrollment_mode: bool = False) -> Tuple[bool, str]:
        """
        Check if face meets quality requirements
        Returns: (is_valid, reason)
        
        enrollment_mode: If True, relaxes quality checks for enrollment photos
        """
        # DEBUG MODE: Bypass all quality checks
        if config.DEBUG_MODE:
            return True, "DEBUG MODE - All checks bypassed"
        
        x, y, w, h = bbox

        # 1. Size check
        if w < self.min_face_size or h < self.min_face_size:
            return False, "Face too small"

        # For enrollment, accept larger faces (high-quality close-ups)
        max_size = 2000 if enrollment_mode else self.max_face_size
        if w > max_size or h > max_size:
            return False, "Face too large (possible poster/photo)"

        # 2. Aspect ratio check (more lenient for enrollment)
        aspect_ratio = w / h if h > 0 else 0
        if enrollment_mode:
            # Accept more angles during enrollment (side poses, etc.)
            if aspect_ratio < 0.5 or aspect_ratio > 1.6:
                return False, "Face distorted"
        else:
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                return False, "Face distorted"

        # 3. Blur detection (more lenient for enrollment)
        blur_threshold = self.blur_threshold * 0.5 if enrollment_mode else self.blur_threshold
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var < blur_threshold:
                return False, f"Face too blurry ({laplacian_var:.1f})"
        except Exception as e:
            logger.debug(f"Blur check failed: {e}")

        # 4. Brightness check (same for both modes)
        try:
            mean_brightness = np.mean(face_img)
            if mean_brightness < 40:
                return False, "Face too dark"
            if mean_brightness > 220:
                return False, "Face too bright"
        except Exception as e:
            logger.debug(f"Brightness check failed: {e}")

        return True, "OK"

    def detect_and_encode_faces(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Simple face detection and encoding for recognition (NO quality validation)
        Returns: List of (embedding, (x, y, w, h)) tuples
        """
        try:
            # Validate frame
            if frame is None or not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
                return []

            # Downscale frame for faster detection if configured
            scale = getattr(config, 'FACE_DETECTION_SCALE', 1.0)
            if scale < 1.0:
                height, width = frame.shape[:2]
                new_width = int(width * scale)
                new_height = int(height * scale)
                small_frame = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                scale_inverse = 1.0 / scale
            else:
                small_frame = frame
                scale_inverse = 1.0

            # Detect faces with InsightFace (no quality filtering)
            faces = self.app.get(small_frame, max_num=getattr(config, 'MAX_FACES_PER_FRAME', 10))

            results = []
            for face in faces:
                # Get bounding box and scale back
                bbox = face.bbox.astype(int)
                x = int(bbox[0] * scale_inverse)
                y = int(bbox[1] * scale_inverse)
                x2 = int(bbox[2] * scale_inverse)
                y2 = int(bbox[3] * scale_inverse)
                w = x2 - x
                h = y2 - y

                # Only check minimum size (no other quality checks)
                min_size = getattr(config, 'MIN_FACE_SIZE', 20)
                if w < min_size or h < min_size:
                    continue

                # Get normalized embedding
                embedding = face.normed_embedding.copy()

                results.append((embedding, (x, y, w, h)))

            # Cleanup
            del faces
            return results

        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []

    def detect_faces(self, frame: np.ndarray, enrollment_mode: bool = False) -> List[Dict]:
        """
        Detect faces using InsightFace ONNX detector with quality validation
        
        enrollment_mode=True: Accept ALL detected faces (no validation)
        enrollment_mode=False: Apply quality checks for enrollment/testing
        """
        try:
            # Detect faces with InsightFace
            faces = self.app.get(frame)

            if len(faces) > 0 and not enrollment_mode:
                logger.info(f"✓ Detected {len(faces)} face(s)")

            valid_faces = []
            for face in faces:
                # Get detection confidence
                det_score = face.det_score
                
                # ENROLLMENT MODE: Accept ALL detected faces (no thresholds)
                if enrollment_mode:
                    # Skip all validation - just get the embedding
                    # Accept ANY detection score (even very low confidence)
                    try:
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Basic bounds check to avoid invalid regions
                        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or w <= 0 or h <= 0:
                            continue
                            
                        face_img = frame[y1:y2, x1:x2]
                        
                        # Skip if face region is empty
                        if face_img.size == 0:
                            continue
                            
                        embedding = face.normed_embedding
                        
                        valid_faces.append({
                            'bbox': [x1, y1, w, h],
                            'embedding': embedding,
                            'det_score': float(det_score),
                            'face_img': face_img
                        })
                    except Exception as e:
                        logger.debug(f"Skipping face due to error: {e}")
                        continue
                    continue
                
                # REAL-TIME MODE: Apply quality checks
                logger.info(f"  Face confidence: {det_score:.2%}")

                if det_score < self.min_detection_score:
                    logger.warning(
                        f"  ✗ Low confidence ({det_score:.2%} < {self.min_detection_score:.2%})")
                    continue

                # Get bbox
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1

                # Extract face region for quality checks
                face_img = frame[y1:y2, x1:x2]

                # Quality checks (only for real-time)
                is_valid, reason = self.check_face_quality(
                    face_img, [x1, y1, w, h], enrollment_mode=False)
                if not is_valid:
                    logger.warning(f"  ✗ Quality check failed: {reason}")
                    continue
                else:
                    logger.info(f"  ✓ Quality check passed")

                # Get normalized embedding (already computed by InsightFace)
                embedding = face.normed_embedding
                embedding = face.normed_embedding

                valid_faces.append({
                    'bbox': [x1, y1, w, h],
                    'embedding': embedding,
                    'det_score': float(det_score),
                    'face_img': face_img
                })

            return valid_faces

        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []

    def recognize_face(self, face_embedding: np.ndarray, threshold: float = None) -> Tuple[Optional[str], Optional[str], float]:
        """
        Recognize a face from its embedding using optimized vector search
        Returns: (employee_id, name, confidence)
        """
        if threshold is None:
            threshold = config.RECOGNITION_THRESHOLD_ONNX

        # DEBUG MODE: Use configured threshold but show detailed info
        if config.DEBUG_MODE:
            logger.info(f"  [DEBUG MODE] Using threshold: {threshold:.2%}")

        if len(self.known_face_encodings) == 0:
            return None, None, 0.0

        try:
            # Normalize query embedding
            face_embedding_normalized = face_embedding / \
                np.linalg.norm(face_embedding)

            # Fast cosine similarity with all embeddings (vectorized)
            similarities = np.dot(self.known_face_encodings,
                                  face_embedding_normalized)

            # Get best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            # DEBUG MODE: Show top 3 matches
            if config.DEBUG_MODE:
                top_indices = np.argsort(similarities)[::-1][:3]
                logger.info(f"  [DEBUG] Top 3 matches:")
                for rank, idx in enumerate(top_indices, 1):
                    emp_id = self.known_employee_ids[idx]
                    emp_name = self.known_employee_names[idx]
                    similarity = similarities[idx]
                    logger.info(f"    {rank}. {emp_name} (ID: {emp_id}): {similarity:.2%}")

            # Log similarity scores for debugging
            logger.info(
                f"  Best match similarity: {best_similarity:.2%} (threshold: {threshold:.2%})")

            # Check if similarity meets threshold (using similarity directly, not distance)
            if best_similarity >= threshold:
                employee_id = self.known_employee_ids[best_idx]
                name = self.known_employee_names[best_idx]
                confidence = best_similarity  # Similarity is our confidence

                logger.info(f"  ✓ Match found: {name} (ID: {employee_id})")
                return employee_id, name, float(confidence)

            logger.warning(
                f"  ✗ No match (best: {best_similarity:.2%} < threshold: {threshold:.2%})")
            return None, None, float(best_similarity)

        except Exception as e:
            logger.error(f"Recognition error: {str(e)}")
            return None, None, 0.0

    def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Recognize faces in frame with cooldown and status management
        Simple detection without quality validation - just detect and match
        Returns: List of dicts with employee info, face location, and status
        """
        logger.debug(f"\n[DEBUG] === Starting face recognition ===")
        logger.debug(f"[DEBUG] Frame shape: {frame.shape}")

        # Detect and encode faces (NO quality validation)
        face_data = self.detect_and_encode_faces(frame)

        if not face_data:
            logger.debug("[DEBUG] No faces detected in frame")
            return []

        recognized_faces = []
        current_time = datetime.now()

        for face_item in face_data:
            try:
                # Validate face_item structure (tuple of embedding and bbox)
                if not isinstance(face_item, (tuple, list)) or len(face_item) != 2:
                    logger.warning(f"Invalid face_item structure: {type(face_item)}, skipping")
                    continue

                embedding, bbox = face_item

                # Validate bbox structure
                if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                    logger.warning(f"Invalid bbox structure: {type(bbox)}, skipping")
                    continue

                x, y, w, h = bbox
                face_width = w
                face_height = h
                face_size = min(face_width, face_height)

                # Convert to (top, right, bottom, left) for compatibility
                face_location = (y, x + w, y + h, x)

                # Check minimum face size
                if face_size < config.MIN_FACE_SIZE:
                    recognized_faces.append({
                        'employee_id': None,
                        'name': 'Too Small',
                        'confidence': 0.0,
                        'face_location': face_location,
                        'bbox': [x, y, w, h],  # For backward compatibility
                        'face_size': face_size,
                        'status': 'too_small',
                        'recognized': False
                    })
                    continue

                # Search for match using vector database
                # DEBUG: Try with relaxed threshold to see all potential matches
                debug_threshold = 1.0 if config.DEBUG_MODE else config.RECOGNITION_THRESHOLD_ONNX
                matches = self.vector_db.search_similar_faces(
                    embedding,
                    n_results=3 if config.DEBUG_MODE else 1,
                    distance_threshold=debug_threshold
                )

                # DEBUG: Show top matches
                if config.DEBUG_MODE and matches:
                    logger.info(f"  [DEBUG] Top {len(matches)} matches:")
                    for idx, m in enumerate(matches, 1):
                        logger.info(f"    {idx}. {m['employee_id']}: distance={m['distance']:.4f}, similarity={1.0-m['distance']:.2%}")
                    logger.info(f"  [DEBUG] Threshold: {config.RECOGNITION_THRESHOLD_ONNX:.4f}")

                if matches and len(matches) > 0:
                    match = matches[0]
                    employee_id = match['employee_id']
                    distance = match['distance']
                    confidence = 1.0 - distance

                    # Check if match meets threshold
                    if distance > config.RECOGNITION_THRESHOLD_ONNX:
                        logger.warning(f"  ✗ Best match distance {distance:.4f} > threshold {config.RECOGNITION_THRESHOLD_ONNX:.4f}")
                        # Treat as unknown
                        recognized_faces.append({
                            'employee_id': None,
                            'name': 'Unknown',
                            'confidence': 0.0,
                            'face_location': face_location,
                            'bbox': [x, y, w, h],
                            'face_size': face_size,
                            'status': 'unknown',
                            'recognized': False
                        })
                        continue

                    # Check cooldown
                    if employee_id in self.recognition_cache:
                        last_recognition = self.recognition_cache[employee_id]
                        cooldown_seconds = getattr(config, 'RECOGNITION_COOLDOWN', 30)
                        if (current_time - last_recognition).total_seconds() < cooldown_seconds:
                            employee = self.db.get_employee_by_id(employee_id)
                            if employee:
                                recognized_faces.append({
                                    'employee_id': employee_id,
                                    'name': employee['name'],
                                    'confidence': confidence,
                                    'face_location': face_location,
                                    'bbox': [x, y, w, h],  # For backward compatibility
                                    'face_size': face_size,
                                    'status': 'cooldown',
                                    'recognized': True
                                })
                            continue

                    # Get employee info
                    employee = self.db.get_employee_by_id(employee_id)
                    if employee:
                        recognized_faces.append({
                            'employee_id': employee_id,
                            'name': employee['name'],
                            'confidence': confidence,
                            'face_location': face_location,
                            'bbox': [x, y, w, h],  # For backward compatibility
                            'face_size': face_size,
                            'status': 'recognized',
                            'recognized': True
                        })

                        # Update cooldown cache
                        self.recognition_cache[employee_id] = current_time
                        logger.info(f"  ✓ Recognized: {employee['name']} ({confidence:.2%})")
                    else:
                        logger.warning(f"Employee {employee_id} not found in database")
                        recognized_faces.append({
                            'employee_id': None,
                            'name': 'Unknown',
                            'confidence': 0.0,
                            'face_location': face_location,
                            'bbox': [x, y, w, h],  # For backward compatibility
                            'face_size': face_size,
                            'status': 'unknown',
                            'recognized': False
                        })
                else:
                    # No match found
                    logger.warning(f"  ✗ Unknown person")
                    recognized_faces.append({
                        'employee_id': None,
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'face_location': face_location,
                        'bbox': [x, y, w, h],  # For backward compatibility
                        'face_size': face_size,
                        'status': 'unknown',
                        'recognized': False
                    })

            except Exception as e:
                logger.error(f"Error processing face: {str(e)}")
                continue

        # Force cleanup of face_data to free memory
        del face_data

        return recognized_faces

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Legacy method for backward compatibility
        Process a frame and return recognized faces
        """
        return self.recognize_faces(frame)

    def draw_annotations(self, frame: np.ndarray, recognized_employees: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame with color-coded status
        """
        annotated = frame.copy()

        for employee in recognized_employees:
            face_location = employee['face_location']
            top, right, bottom, left = face_location

            # Color coding by status
            if employee['status'] == 'recognized':
                color = (0, 255, 0)  # Green
                label = f"{employee['name']} ({employee['confidence']:.2f})"
            elif employee['status'] == 'cooldown':
                color = (0, 165, 255)  # Orange
                label = f"{employee['name']} (Already Marked)"
            elif employee['status'] == 'spoofing':
                color = (0, 0, 255)  # Red
                label = "SPOOFING DETECTED!"
            elif employee['status'] == 'too_small':
                color = (255, 0, 255)  # Magenta
                label = f"Too Small ({employee['face_size']}px)"
            else:  # unknown
                color = (255, 255, 0)  # Cyan
                label = "Unknown Person"

            # Draw rectangle
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

            # Draw label background
            cv2.rectangle(annotated, (left, bottom - 35),
                          (right, bottom), color, cv2.FILLED)

            # Draw label text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(annotated, label, (left + 6, bottom - 6),
                        font, 0.6, (255, 255, 255), 1)

        return annotated

    def check_cooldown(self, employee_id: str) -> bool:
        """
        Check if employee is in cooldown period
        Returns True if cooldown has expired (can recognize again)
        Returns False if still in cooldown
        """
        if employee_id not in self.recognition_cache:
            return True

        last_recognition = self.recognition_cache[employee_id]
        cooldown_seconds = getattr(config, 'RECOGNITION_COOLDOWN', 30)
        elapsed = (datetime.now() - last_recognition).total_seconds()

        return elapsed >= cooldown_seconds

    def update_cooldown(self, employee_id: str):
        """Update the cooldown timestamp for an employee"""
        self.recognition_cache[employee_id] = datetime.now()

    def save_captured_face(self, frame: np.ndarray, employee_id: str) -> Optional[str]:
        """
        Save captured face image for audit/logging purposes
        Returns: path to saved image or None if saving failed
        """
        save_faces = getattr(config, 'SAVE_CAPTURED_FACES', False)
        if not save_faces:
            return None

        try:
            # Create captured faces directory if it doesn't exist
            captured_dir = getattr(config, 'CAPTURED_FACES_DIR', 'captured_faces')
            os.makedirs(captured_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{employee_id}_{timestamp}.jpg"
            filepath = os.path.join(captured_dir, filename)

            # Save frame
            cv2.imwrite(filepath, frame)
            logger.info(f"Saved captured face: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving captured face: {str(e)}")
            return None

    def encode_face_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate face encoding from image file (for enrollment)
        Returns: normalized embedding or None if failed
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None

            # Detect faces in enrollment mode
            faces = self.detect_faces(img, enrollment_mode=True)

            if not faces or len(faces) == 0:
                logger.error(f"No face detected in: {image_path}")
                return None

            if len(faces) > 1:
                logger.warning(f"Multiple faces detected, using first face")

            embedding = faces[0]['embedding']
            logger.info(f"✓ Generated encoding: {len(embedding)}D vector")
            return embedding

        except Exception as e:
            logger.error(f"Error encoding face: {str(e)}")
            return None

    def add_employee_encoding(self, employee_id: str, name: str, encoding: np.ndarray) -> bool:
        """
        Add new employee encoding to the system
        Returns: True if successful, False otherwise
        """
        try:
            success = self.vector_db.add_face_encoding(
                employee_id=employee_id,
                encoding=encoding,
                metadata={'name': name}
            )

            if success:
                # Reload encodings to include new employee
                self.load_employee_encodings()
                logger.info(f"✓ Added encoding for {name} (ID: {employee_id})")
                return True
            else:
                logger.error(f"Failed to add encoding for {name}")
                return False

        except Exception as e:
            logger.error(f"Error adding employee encoding: {str(e)}")
            return False

    def update_employee_encoding(self, employee_id: str, new_encoding: np.ndarray) -> bool:
        """
        Update existing employee encoding
        Returns: True if successful, False otherwise
        """
        try:
            success = self.vector_db.update_face_encoding(
                employee_id, new_encoding)

            if success:
                # Reload encodings to reflect update
                self.load_employee_encodings()
                logger.info(f"✓ Updated encoding for employee ID: {employee_id}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error updating encoding: {str(e)}")
            return False

    def remove_employee_encoding(self, employee_id: str) -> bool:
        """
        Remove employee encoding from the system
        Returns: True if successful, False otherwise
        """
        try:
            success = self.vector_db.delete_face_encoding(employee_id)

            if success:
                # Reload encodings to reflect removal
                self.load_employee_encodings()
                logger.info(f"✓ Removed encoding for employee ID: {employee_id}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error removing encoding: {str(e)}")
            return False

    def enroll_face(self, frame: np.ndarray, employee_id: str, name: str) -> Tuple[bool, str]:
        """
        Enroll a new face or update existing one
        Returns: (success, message)
        """
        try:
            # Detect faces in enrollment mode (lenient)
            faces = self.detect_faces(frame, enrollment_mode=True)

            if len(faces) == 0:
                return False, "No face detected in image"

            if len(faces) > 1:
                return False, "Multiple faces detected, please provide image with single face"

            face_data = faces[0]
            embedding = face_data['embedding']

            # Store in vector database
            metadata = {
                'name': name,
                'enrolled_date': datetime.now().isoformat()
            }

            self.vector_db.add_embedding(employee_id, embedding, metadata)

            # Reload encodings
            self.load_employee_encodings()

            return True, f"Successfully enrolled {name}"

        except Exception as e:
            logger.error(f"Enrollment error: {str(e)}")
            return False, f"Enrollment failed: {str(e)}"

    def cleanup(self):
        """
        Clean up resources and free memory
        """
        try:
            # Clear recognition cache
            self.recognition_cache.clear()

            # Clear face encodings from memory
            self.known_face_encodings.clear()
            self.known_employee_ids.clear()
            self.known_employee_names.clear()

            # Close databases
            if self.db:
                self.db.close()
            if self.vector_db:
                self.vector_db.close()

            # Clean up InsightFace app
            if self.app:
                del self.app
                self.app = None

            # Force garbage collection
            import gc
            gc.collect()

            logger.info("✓ Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
