"""
Configuration settings for the Kramer API.
"""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Kramer Discovery Platform"
    VERSION: str = "1.0.0"

    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Data Paths
    DATA_DIR: str = "../data"
    OUTPUTS_DIR: str = "../outputs"
    UPLOADS_DIR: str = "../data/uploads"

    # Database
    DATABASE_PATH: str = "../data/discoveries.db"

    # API Keys (loaded from environment)
    ANTHROPIC_API_KEY: str = ""

    class Config:
        env_file = "../.env"
        case_sensitive = True


settings = Settings()
