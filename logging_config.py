"""
Production Logging Configuration
- File-based logging with rotation
- Colored console output
- Structured logging for monitoring
- Different levels for different components
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
import config

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logging(log_dir: str = None, log_level: str = 'INFO'):
    """
    Setup comprehensive logging system

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if log_dir is None:
        log_dir = config.LOGS_DIR

    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # 1. Console Handler with colors (if available)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if HAS_COLORLOG:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 2. Main log file with rotation (general logs)
    main_log_file = os.path.join(log_dir, 'attendance_system.log')
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 3. Error log file (only errors and critical)
    error_log_file = os.path.join(log_dir, 'errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=10,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)

    # 4. Recognition log file (for audit trail)
    recognition_log_file = os.path.join(log_dir, 'recognition.log')
    recognition_handler = logging.handlers.RotatingFileHandler(
        recognition_log_file,
        maxBytes=20*1024*1024,  # 20MB
        backupCount=10,
        encoding='utf-8'
    )
    recognition_handler.setLevel(logging.INFO)
    recognition_handler.setFormatter(file_formatter)

    # Create recognition logger
    recognition_logger = logging.getLogger('recognition')
    recognition_logger.addHandler(recognition_handler)
    recognition_logger.propagate = False

    # 5. Performance log file
    performance_log_file = os.path.join(log_dir, 'performance.log')
    performance_handler = logging.handlers.RotatingFileHandler(
        performance_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    performance_handler.setLevel(logging.DEBUG)
    performance_handler.setFormatter(file_formatter)

    performance_logger = logging.getLogger('performance')
    performance_logger.addHandler(performance_handler)
    performance_logger.propagate = False

    # Reduce noise from external libraries
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('insightface').setLevel(logging.WARNING)
    logging.getLogger('onnxruntime').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Log startup message
    root_logger.info("=" * 60)
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log directory: {log_dir}")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_recognition_event(employee_id: str, name: str, confidence: float,
                          action: str, success: bool, camera: str = None,
                          liveness_score: float = None):
    """
    Log a recognition event for audit trail

    Args:
        employee_id: Employee identifier
        name: Employee name
        confidence: Recognition confidence
        action: Action type (check_in, check_out)
        success: Whether attendance was marked
        camera: Camera name
        liveness_score: Liveness detection score
    """
    logger = logging.getLogger('recognition')

    msg_parts = [
        f"Employee: {employee_id} ({name})",
        f"Confidence: {confidence:.2%}",
        f"Action: {action}",
        f"Success: {success}"
    ]

    if camera:
        msg_parts.append(f"Camera: {camera}")

    if liveness_score is not None:
        msg_parts.append(f"Liveness: {liveness_score:.2f}")

    message = " | ".join(msg_parts)

    if success:
        logger.info(message)
    else:
        logger.warning(message)


def log_performance(component: str, operation: str, duration_ms: float,
                    metadata: dict = None):
    """
    Log performance metrics

    Args:
        component: Component name (e.g., 'face_detection', 'recognition')
        operation: Operation description
        duration_ms: Duration in milliseconds
        metadata: Additional metadata
    """
    logger = logging.getLogger('performance')

    msg_parts = [
        f"Component: {component}",
        f"Operation: {operation}",
        f"Duration: {duration_ms:.2f}ms"
    ]

    if metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        msg_parts.append(f"Metadata: {meta_str}")

    message = " | ".join(msg_parts)
    logger.debug(message)


def log_camera_status(camera_name: str, status: str, details: str = None):
    """
    Log camera status changes

    Args:
        camera_name: Camera identifier
        status: Status (connected, disconnected, error, etc.)
        details: Additional details
    """
    logger = logging.getLogger('camera')

    message = f"Camera: {camera_name} | Status: {status}"
    if details:
        message += f" | Details: {details}"

    if status in ['connected', 'running']:
        logger.info(message)
    elif status in ['disconnected', 'reconnecting']:
        logger.warning(message)
    else:
        logger.error(message)


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, component: str, operation: str, metadata: dict = None):
        self.component = component
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() -
                        self.start_time).total_seconds() * 1000
            log_performance(self.component, self.operation,
                            duration, self.metadata)
