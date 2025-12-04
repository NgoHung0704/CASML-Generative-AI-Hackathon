"""
Utility functions - Logging, seeding, and helper functions
"""

import os
import random
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime


def setup_logging(config):
    """
    Set up logging configuration.
    
    Args:
        config: Configuration object
    """
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.log_file', 'logs/experiment.log')
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Set deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_gpu(memory_growth: bool = True, memory_limit: int = None):
    """
    Configure TensorFlow GPU settings.
    
    Args:
        memory_growth: Enable GPU memory growth
        memory_limit: GPU memory limit in MB (None for no limit)
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            if memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit
            if memory_limit:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


def get_device(preferred_device: str = 'GPU:0') -> str:
    """
    Get the best available device for TensorFlow.
    
    Args:
        preferred_device: Preferred device ('GPU:0', 'GPU:1', 'CPU')
    
    Returns:
        Device string
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if 'GPU' in preferred_device and gpus:
        return preferred_device
    else:
        return 'CPU'


def save_results(results: dict, output_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    import json
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(input_path: str) -> dict:
    """
    Load results from JSON file.
    
    Args:
        input_path: Input file path
    
    Returns:
        Results dictionary
    """
    import json
    
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Progress description
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            print(f"\r{self.description}: {self.current}/{self.total} "
                  f"({100*self.current/self.total:.1f}%) "
                  f"[{format_time(elapsed)} < {format_time(eta)}]", end='')
    
    def close(self):
        """Close progress tracker."""
        print()  # New line
