"""
Centralized logging configuration for the Deduper application.
Provides uniform logging across all modules with proper formatting and levels.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from flask import request


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    log_level: str = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Set up centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        enable_console: Whether to enable console logging
        enable_colors: Whether to use colored console output
    
    Returns:
        Configured logger instance
    """
    # Get log level from environment or parameter
    level = log_level or os.environ.get('DEDUPER_LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, level, logging.INFO)
    
    # Create root logger
    logger = logging.getLogger('deduper')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if enable_colors and sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter(simple_formatter._fmt, simple_formatter.datefmt))
        else:
            console_handler.setFormatter(simple_formatter)
        
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f'deduper.{name}')


def configure_flask_logging(app):
    """
    Configure Flask application logging.
    
    Args:
        app: Flask application instance
    """
    # Set Flask log level
    app.logger.setLevel(logging.INFO)
    
    # Remove default Flask handlers to avoid duplicate logs
    for handler in list(app.logger.handlers):
        app.logger.removeHandler(handler)
    
    # Add our custom handler
    app.logger.addHandler(logging.getLogger('deduper').handlers[0])
    
    # Configure Werkzeug (Flask's WSGI server) logging
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)  # Reduce noise from Werkzeug
    
    # Add request logging
    @app.before_request
    def log_request_info():
        logger = get_logger('flask.requests')
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    
    @app.after_request
    def log_response_info(response):
        logger = get_logger('flask.requests')
        logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
        return response


# Global logger instance
logger = setup_logging()

