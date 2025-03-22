import os

class Config:
    """Base configuration."""
    API_URL = os.getenv("API_URL", "http://api:8000")
    DEBUG = False
    TESTING = False
    
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    
class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

# Default configuration
DefaultConfig = DevelopmentConfig