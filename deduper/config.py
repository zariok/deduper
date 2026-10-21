import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

_logger = logging.getLogger(__name__)


def env_positive_int(name: str, default: int) -> int:
    """Read a positive integer from environment variable *name*.

    Unset or blank yields *default*. A value that is not an integer, or is zero
    or negative, also yields *default* - with a warning, so a typo in a unit file
    degrades to the stock pool size instead of stopping the app.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        _logger.warning("Ignoring %s=%r: not an integer; using %d", name, raw, default)
        return default
    if value < 1:
        _logger.warning("Ignoring %s=%r: must be positive; using %d", name, raw, default)
        return default
    return value


def default_hash_workers() -> int:
    """Half the cores, at least 4. Hashing is decode-bound and each worker holds
    a decoded image, so one per hardware thread buys little and costs memory."""
    return max(4, (os.cpu_count() or 8) // 2)


class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        # Only allow 'dev' key in development mode
        if os.environ.get('FLASK_ENV') == 'development' or os.environ.get('DEDUPER_DEV', '').lower() == 'true':
            SECRET_KEY = 'dev'
            import warnings
            warnings.warn("Using insecure 'dev' SECRET_KEY. Set SECRET_KEY environment variable in production!")
        else:
            print("ERROR: SECRET_KEY environment variable must be set in production!", file=sys.stderr)
            sys.exit(1)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Server settings
    PORT = int(os.environ.get('DEDUPER_PORT', 5000))
    HOST = os.environ.get('DEDUPER_HOST', '127.0.0.1')
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = Path(os.environ.get('DEDUPER_DATA_DIR', BASE_DIR / 'data'))
    TEMPLATES_DIR = BASE_DIR / 'templates'
    STATIC_DIR = BASE_DIR / 'static'
    
    # Worker pool sizes. These used to be hard-coded (hash pool capped at 8 for
    # memory pressure on a small box, 4 ffmpeg threads, 2 scan workers), which
    # left most of the production host (80 cores, 755 GB RAM) idle. Each scanned
    # folder has its own SQLite database, so concurrent scan workers do not
    # contend on a shared cache.
    HASH_WORKERS = env_positive_int('DEDUPER_HASH_WORKERS', default_hash_workers())
    FFMPEG_WORKERS = env_positive_int('DEDUPER_FFMPEG_WORKERS', 8)
    SCAN_WORKERS = env_positive_int('DEDUPER_SCAN_WORKERS', 4)

    # Supported file extensions
    # NOTE: .gif is classified as video because animated GIFs are temporal media.
    # Hashing only the first frame (image pipeline) causes false matches with static images.
    # FFmpeg handles GIFs as video streams, extracting a consistent thumbnail frame.
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.avif'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.gif', '.webm', '.flv', '.wmv', '.m4v'}
    
    @classmethod
    def init_app(cls, app: "Flask") -> None:
        """Initialize the application configuration."""
        # Create necessary directories with proper permissions
        for directory in [cls.DATA_DIR, cls.TEMPLATES_DIR, cls.STATIC_DIR]:
            directory.mkdir(exist_ok=True, parents=True)
            # Ensure the directory is readable and writable
            os.chmod(directory, 0o755)

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    DATA_DIR = Path(__file__).parent.parent / 'tests' / 'test_data'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 