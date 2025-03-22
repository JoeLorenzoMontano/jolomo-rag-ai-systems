from flask import Flask
import logging
import os

def create_app(config_object=None):
    """Create and configure the Flask application.
    
    Args:
        config_object: Configuration object or string. If None, uses DefaultConfig.
        
    Returns:
        Flask: Configured Flask application
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_object is None:
        from ui.config import DefaultConfig
        app.config.from_object(DefaultConfig)
    else:
        app.config.from_object(config_object)
    
    # Log the API URL being used
    logging.info(f"Using API URL: {app.config['API_URL']}")
    
    # Register blueprints
    register_blueprints(app)
    
    return app

def register_blueprints(app):
    """Register all blueprint modules with the application.
    
    Args:
        app (Flask): Flask application
    """
    from ui.routes.main import main_bp
    from ui.routes.documents import documents_bp
    from ui.routes.query import query_bp
    from ui.routes.system import system_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(query_bp)
    app.register_blueprint(system_bp)