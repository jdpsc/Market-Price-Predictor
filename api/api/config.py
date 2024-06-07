import enum
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    # General configurations.
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: LogLevel = LogLevel.INFO
    # - Current version of the API.
    VERSION: str = "v1"
    # - Quantity of workers for uvicorn.
    WORKERS_COUNT: int = 1
    # - Enable uvicorn reloading.
    RELOAD: bool = False

    PROJECT_NAME: str = "Market Price Predictor API"

    # AWS configurations
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_PREDS_BUCKET: Optional[str] = None

    # API key
    KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "API_"
        case_sensitive = False
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()
