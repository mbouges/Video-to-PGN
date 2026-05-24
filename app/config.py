"""Application configuration using Pydantic settings."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    gemini_api_key: str = ""

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    temp_dir: Path = Path(__file__).parent.parent / "temp"
    output_dir: Path = Path(__file__).parent.parent / "output"

    # Video download settings
    video_quality: str = "720p"
    max_video_duration_seconds: int = 7200  # 2 hours

    # Frame sampling settings
    sample_interval_seconds: float = 3.0  # one frame every N seconds
    dedup_threshold: int = 5  # perceptual hash hamming distance for dedup

    # Gemini Vision settings
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_retries: int = 3
    gemini_retry_delay: float = 2.0
    api_call_delay: float = 1.0  # seconds between API calls

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "VTP_", "env_file": ".env", "extra": "ignore"}

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
