"""
config.py — App settings via Pydantic
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "DermAI"
    DEBUG: bool = False

    # CORS — allows the React frontend to call the API
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # Model path — uses baseline (best result so far: 0.769 AUC)
    MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models", "ham10000_baseline.keras"
    )

    # Upload limits
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]

    class Config:
        env_file = ".env"

settings = Settings()
