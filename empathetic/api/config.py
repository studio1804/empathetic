"""Configuration for the Empathetic API."""
from functools import lru_cache
from typing import List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from ..config import config as app_config


class Settings(BaseSettings):
    """API configuration settings."""
    
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore"  # Ignore extra environment variables
    )
    
    VERSION: str = "0.1.0"
    DEBUG: bool = app_config.debug
    
    DATABASE_URL: str = app_config.database_url
    SECRET_KEY: str = app_config.secret_key
    
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8000",
    ]
    
    OPENAI_API_KEY: str = app_config.openai_api_key
    ANTHROPIC_API_KEY: str = app_config.anthropic_api_key


@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return Settings()