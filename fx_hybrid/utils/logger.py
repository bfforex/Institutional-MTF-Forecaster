"""
Logging Utilities
"""

import logging
import sys
from typing import Optional
import os
from datetime import datetime


def setup_logger(
    name: str = 'fx_hybrid',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file
    level : int
        Logging level
    console : bool
        Whether to log to console
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'fx_hybrid') -> logging.Logger:
    """
    Get existing logger or create default.
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class LoggerContext:
    """Context manager for temporary log level change."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_experiment(
    logger: logging.Logger,
    config: dict,
    metrics: dict,
    model_info: Optional[dict] = None
):
    """
    Log experiment details.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    config : dict
        Configuration used
    metrics : dict
        Results metrics
    model_info : dict, optional
        Model information
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT LOG")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    if model_info:
        logger.info("MODEL INFO:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        logger.info("")
    
    logger.info("CONFIGURATION:")
    _log_dict(logger, config, indent=2)
    logger.info("")
    
    logger.info("RESULTS:")
    _log_dict(logger, metrics, indent=2)
    logger.info("=" * 70)


def _log_dict(logger: logging.Logger, d: dict, indent: int = 0):
    """Recursively log dictionary."""
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info(" " * indent + f"{key}:")
            _log_dict(logger, value, indent + 2)
        else:
            logger.info(" " * indent + f"{key}: {value}")
