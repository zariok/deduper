import sys
from flask import Flask, request
from .config import config
from .utils.media import check_all_requirements
from .utils.logging_config import setup_logging, configure_flask_logging, get_logger
from .routes.views import bp

# Set up logging before creating the app
logger = setup_logging()

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
    logger.info("Flask application created successfully")
    
    return app 