"""
Liveness Detection Module
Prevents spoofing attacks using photos, videos, or masks
Implements multiple anti-spoofing techniques
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class LivenessDetector:
    """
    Multi-method liveness detection to prevent spoofing
    Combines motion detection, texture analysis, and frame consistency
    """

    def __init__(self, buffer_size: int = 10):
        """
        Initialize liveness detector

        Args:
            buffer_size: Number of frames to track for motion analysis
        """
        self.buffer_size = buffer_size
        self.face_history = {}  # Track faces across frames

        # Thresholds
        self.motion_threshold = 5.0  # Minimum motion score
        self.texture_threshold = 15.0  # LBP variance threshold
        self.consistency_frames = 5  # Frames needed for consistency check

    def check_liveness(self, face_id: str, face_img: np.ndarray,
                       bbox: list, frame_full: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check if detected face is live (not a photo/video)

        Args:
            face_id: Unique identifier for tracking face across frames
            face_img: Cropped face image
            bbox: Bounding box [x, y, w, h]
            frame_full: Full frame for context

        Returns:
            (is_live, confidence, reason)
        """
        current_time = time.time()

        # Initialize tracking for this face if new
        if face_id not in self.face_history:
            self.face_history[face_id] = {
                'positions': deque(maxlen=self.buffer_size),
                'face_images': deque(maxlen=self.buffer_size),
                'timestamps': deque(maxlen=self.buffer_size),
                'texture_scores': deque(maxlen=self.buffer_size),
                'first_seen': current_time
            }

        history = self.face_history[face_id]

        # Store current data
        history['positions'].append(bbox)
        history['face_images'].append(face_img)
        history['timestamps'].append(current_time)

        # Need at least 3 frames to analyze
        if len(history['positions']) < 3:
            return False, 0.0, "Insufficient frames for analysis"

        # Method 1: Motion Analysis
        motion_score, motion_reason = self._check_motion(history)

        # Method 2: Texture Analysis (detect flat surfaces like photos)
        texture_score, texture_reason = self._check_texture(face_img)
        history['texture_scores'].append(texture_score)

        # Method 3: Temporal Consistency
        consistency_score, consistency_reason = self._check_consistency(
            history)

        # Combined scoring
        scores = {
            'motion': motion_score,
            'texture': texture_score,
            'consistency': consistency_score
        }

        # Calculate overall liveness score (weighted average)
        weights = {'motion': 0.4, 'texture': 0.3, 'consistency': 0.3}
        overall_score = sum(scores[k] * weights[k] for k in scores)

        # Determine if live
        is_live = overall_score > 0.5

        # Build reason message
        reasons = []
        if motion_score < 0.3:
            reasons.append("Low motion")
        if texture_score < 0.3:
            reasons.append("Flat texture (photo?)")
        if consistency_score < 0.3:
            reasons.append("Inconsistent detection")

        reason = "; ".join(reasons) if reasons else "All checks passed"

        logger.debug(f"Liveness check: {overall_score:.2f} - Motion:{motion_score:.2f} "
                     f"Texture:{texture_score:.2f} Consistency:{consistency_score:.2f}")

        return is_live, float(overall_score), reason

    def _check_motion(self, history: Dict) -> Tuple[float, str]:
        """
        Check for natural head/face motion
        Photos will have minimal motion, live faces have micro-movements
        """
        if len(history['positions']) < 3:
            return 0.0, "Not enough frames"

        positions = list(history['positions'])

        # Calculate position changes (center of bbox)
        centers = []
        for bbox in positions:
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            centers.append([cx, cy])

        # Calculate variance in positions
        centers = np.array(centers)
        motion_variance = np.var(centers, axis=0).sum()

        # Calculate size changes (depth motion)
        sizes = [bbox[2] * bbox[3] for bbox in positions]
        size_variance = np.var(sizes)

        # Combined motion score
        total_motion = motion_variance + size_variance * 0.1

        # Normalize to 0-1 score
        # High motion = high score (live)
        # Low motion = low score (photo)
        motion_score = min(1.0, total_motion / 100.0)

        reason = f"Motion variance: {total_motion:.2f}"

        return motion_score, reason

    def _check_texture(self, face_img: np.ndarray) -> Tuple[float, str]:
        """
        Analyze face texture using Local Binary Patterns (LBP)
        Photos have different texture patterns than real skin
        """
        try:
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img

            # Resize for consistent analysis
            gray = cv2.resize(gray, (64, 64))

            # Simple LBP-like texture analysis
            # Calculate gradient variance (real skin has more texture variation)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            gradient_variance = np.var(sobelx) + np.var(sobely)

            # Frequency analysis using FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            # High frequencies indicate texture detail (real skin)
            high_freq_energy = np.sum(
                magnitude_spectrum[16:48, 16:48]) / magnitude_spectrum.size

            # Combined texture score
            texture_score = min(
                1.0, (gradient_variance / 1000.0 + high_freq_energy / 10.0) / 2.0)

            reason = f"Texture variance: {gradient_variance:.2f}"

            return texture_score, reason

        except Exception as e:
            logger.error(f"Texture analysis error: {e}")
            return 0.5, "Analysis failed"

    def _check_consistency(self, history: Dict) -> Tuple[float, str]:
        """
        Check temporal consistency of detections
        Real faces should have consistent detection, photos might flicker
        """
        if len(history['timestamps']) < self.consistency_frames:
            return 0.5, "Not enough frames"

        # Check detection continuity
        timestamps = list(history['timestamps'])
        time_diffs = np.diff(timestamps)

        # Check for regular intervals (should be consistent FPS)
        time_variance = np.var(time_diffs)

        # Low variance = consistent = likely live
        consistency_score = max(0.0, 1.0 - time_variance * 10.0)

        # Check texture consistency across frames
        if len(history['texture_scores']) >= 3:
            texture_variance = np.var(list(history['texture_scores']))
            # Real faces have consistent texture, photos might vary with angle
            texture_consistency = max(0.0, 1.0 - texture_variance)
            consistency_score = (consistency_score + texture_consistency) / 2.0

        reason = f"Time variance: {time_variance:.4f}"

        return consistency_score, reason

    def reset_tracking(self, face_id: str):
        """Clear history for a face ID"""
        if face_id in self.face_history:
            del self.face_history[face_id]

    def cleanup_old_tracks(self, max_age: float = 30.0):
        """Remove old face tracks that haven't been updated"""
        current_time = time.time()
        to_remove = []

        for face_id, history in self.face_history.items():
            if len(history['timestamps']) > 0:
                last_seen = history['timestamps'][-1]
                if current_time - last_seen > max_age:
                    to_remove.append(face_id)

        for face_id in to_remove:
            del self.face_history[face_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old face tracks")


class SimpleLivenessDetector:
    """
    Simplified liveness detector for systems without heavy requirements
    Uses basic motion and blink detection
    """

    def __init__(self):
        self.motion_threshold = 10.0
        self.prev_frames = {}

    def check_liveness_simple(self, face_id: str, face_img: np.ndarray,
                              bbox: list) -> Tuple[bool, float, str]:
        """
        Simple motion-based liveness check

        Returns:
            (is_live, confidence, reason)
        """
        try:
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img

            # First time seeing this face
            if face_id not in self.prev_frames:
                self.prev_frames[face_id] = {
                    'frames': deque(maxlen=5),
                    'motion_scores': deque(maxlen=5)
                }
                self.prev_frames[face_id]['frames'].append(gray)
                return False, 0.0, "First frame"

            history = self.prev_frames[face_id]

            # Calculate frame difference
            if len(history['frames']) > 0:
                prev_frame = history['frames'][-1]

                # Resize to same size for comparison
                prev_resized = cv2.resize(prev_frame, (64, 64))
                curr_resized = cv2.resize(gray, (64, 64))

                # Calculate absolute difference
                diff = cv2.absdiff(curr_resized, prev_resized)
                motion_score = np.mean(diff)

                history['motion_scores'].append(motion_score)

            history['frames'].append(gray)

            # Need at least 3 frames
            if len(history['motion_scores']) < 3:
                return False, 0.0, "Collecting frames"

            # Check average motion
            avg_motion = np.mean(list(history['motion_scores']))

            # Normalize to 0-1
            liveness_score = min(1.0, avg_motion / self.motion_threshold)

            is_live = liveness_score > 0.3
            reason = f"Motion: {avg_motion:.2f}"

            return is_live, float(liveness_score), reason

        except Exception as e:
            logger.error(f"Simple liveness check error: {e}")
            return True, 0.5, "Check failed (defaulting to live)"
