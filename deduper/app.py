import os
import signal
import sys
import atexit
from flask import Flask
from flask_socketio import SocketIO
from .config import config, Config
from .utils.media import check_all_requirements
from .utils.logging_config import setup_logging, configure_flask_logging, get_logger
from .routes.views import bp
from .services.background_scanner import init_background_scanner, stop_background_scanner

# Set up logging before creating the app
logger = setup_logging()

# Module-level SocketIO instance so it can be imported by __main__.py
socketio = SocketIO()


def create_app(config_name='default'):
    """Create and configure the Flask application."""
    logger.info(f"Creating Flask application with config: {config_name}")

    # Check all application requirements before creating the Flask app
    requirements_met, error_message = check_all_requirements()
    if not requirements_met:
        print(error_message)  # Print to console so user sees it
        logger.error("Application requirements not met, exiting application")
        sys.exit(1)

    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static')

    # Load configuration
    app.config.from_object(config[config_name])
    logger.info(f"Loaded configuration: {config_name}")

    # Initialize the application
    config[config_name].init_app(app)

    # Configure Flask logging
    configure_flask_logging(app)

    # Register blueprints
    app.register_blueprint(bp)

    # Initialize Socket.IO (Phase 3.2)
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # Register WebSocket event handlers
    from .routes.socketio_events import register_socketio_events, init_socketio_ref
    register_socketio_events(socketio)
    init_socketio_ref(socketio)

    # Start background scanner (only in non-testing mode)
    if config_name != 'testing':
        _start_background_scanner(app)

    logger.info("Flask application created successfully")

    return app


_cleanup_done = False


def _cleanup():
    """Shut down background scanner, process pool, and SQLite caches."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    logger.info("Cleaning up resources...")
    try:
        stop_background_scanner()
    except Exception:
        pass
    try:
        from .services.duplicate_finder import shutdown_process_pool
        shutdown_process_pool()
    except Exception:
        pass
    try:
        from .utils.hash_cache import close_all_caches
        close_all_caches()
    except Exception:
        pass
    logger.info("Cleanup complete")


def _start_background_scanner(app: Flask):
    """Start the background scanner service."""
    # When Flask runs with debug=True and use_reloader=True (the default),
    # it spawns two processes: a parent reloader process and a child worker.
    # Only start the background scanner in the worker process.
    # The worker process has WERKZEUG_RUN_MAIN set to "true".
    if app.debug and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        logger.debug("Skipping background scanner in reloader process")
        return

    try:
        data_dir = str(Config.DATA_DIR)
        image_extensions = Config.IMAGE_EXTENSIONS
        video_extensions = Config.VIDEO_EXTENSIONS

        scanner = init_background_scanner(
            data_dir=data_dir,
            image_extensions=image_extensions,
            video_extensions=video_extensions
        )
        logger.info("Background scanner initialized and started")

        # Register cleanup on app shutdown
        atexit.register(_cleanup)

        # Handle SIGINT/SIGTERM so child processes are cleaned up on Ctrl+C
        def _signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            _cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    except Exception as e:
        logger.error(f"Failed to start background scanner: {e}", exc_info=True)
        # Don't fail app startup if background scanner fails
