"""Centralized application configuration."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration."""

    app_name: str = "Multimodal AI Agent API"
    app_environment: str = "development"
    log_level: str = "INFO"

    # Read credentials and model configuration from environment variables.
    ollama_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OLLAMA_API_KEY",
    )

    # Text-generation model.
    ollama_model: str = Field(
        default="gemma4:31b",
        min_length=1,
        validation_alias="OLLAMA_MODEL",
    )

    # Vision model, configured independently from the text model.
    ollama_vision_model: str = Field(
        default="gemma4:31b",
        min_length=1,
        validation_alias="OLLAMA_VISION_MODEL",
    )

    ollama_base_url: str = Field(
        default="https://ollama.com/api",
        validation_alias="OLLAMA_BASE_URL",
    )

    ollama_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
        validation_alias="OLLAMA_TIMEOUT_SECONDS",
    )

    # Local speech-to-text model configuration.
    whisper_model: str = Field(
        default="base",
        min_length=1,
        validation_alias="WHISPER_MODEL",
    )

    whisper_device: str = Field(
        default="cpu",
        min_length=1,
        validation_alias="WHISPER_DEVICE",
    )

    whisper_compute_type: str = Field(
        default="int8",
        min_length=1,
        validation_alias="WHISPER_COMPUTE_TYPE",
    )

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


# Shared settings instance used by application services.
settings = get_settings()
