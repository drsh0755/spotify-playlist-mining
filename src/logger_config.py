"""
Centralized logging configuration for all scripts
Logs to both console and file with timestamps
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(script_name, log_dir="logs"):
    """
    Set up logger that writes to both console and file
    
    Args:
        script_name: Name of the script (used for log filename)
        log_dir: Directory to store log files
    
    Returns:
        logger: Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"{script_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - less verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log the log file location
    logger.info(f"Logging to: {log_file}")
    
    return logger

def log_section(logger, title):
    """Print a formatted section header"""
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)

def log_subsection(logger, title):
    """Print a formatted subsection header"""
    logger.info("-" * 70)
    logger.info(title)
    logger.info("-" * 70)
