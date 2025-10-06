import os
from pathlib import Path

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Server settings
    PORT = int(os.environ.get('DEDUPER_PORT', 5000))
    HOST = os.environ.get('DEDUPER_HOST', '127.0.0.1')
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = Path(os.environ.get('DEDUPER_DATA_DIR', BASE_DIR / 'data'))
    TEMPLATES_DIR = BASE_DIR / 'templates'
    STATIC_DIR = BASE_DIR / 'static'
    
    # Supported file extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
    
    @classmethod
    def init_app(cls, app):
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