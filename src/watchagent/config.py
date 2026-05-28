"""Application configuration.

Every tunable in the system is declared here so it can be overridden via
environment variables (or a local `.env` file) without code changes.
The README references these names directly when justifying detector thresholds.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Polling ----------------------------------------------------------
    poll_interval_seconds: int = Field(600, gt=0)
    http_timeout_seconds: float = Field(10.0, gt=0)
    max_retries: int = Field(3, ge=0)
    retry_backoff_base_seconds: float = Field(1.0, gt=0)

    # --- Storage ----------------------------------------------------------
    # Local-friendly default so `pytest` and ad-hoc CLI runs work without
    # a `/data` directory. docker-compose.yml overrides this to
    # `/data/watchagent.db` (the named-volume mount point) via the service's
    # `environment:` block — those keys win over .env, so the same .env
    # works for both local dev and the container.
    db_path: str = "watchagent.db"

    # --- Logging ----------------------------------------------------------
    log_level: str = Field("INFO")

    # --- Detection: per-city temperature anomaly (z-score) ---------------
    w: int = Field(48, gt=1, description="rolling window size per city")
    min_samples: int = Field(6, ge=2, description="warm-up sample threshold")
    z_thresh: float = Field(2.5, gt=0)

    # --- Detection: rapid rate-of-change ---------------------------------
    rate_thresh: float = Field(4.0, gt=0, description="°C / hour")

    # --- Detection: wind danger (absolute) -------------------------------
    wind_thresh: float = Field(40.0, gt=0, description="km/h sustained")

    # --- Detection: apparent-vs-actual divergence ------------------------
    divergence_thresh: float = Field(5.0, gt=0, description="°C")

    # --- Debounce / cooldown ---------------------------------------------
    cooldown_seconds: int = Field(10800, ge=0, description="per (city, event_type)")

    @model_validator(mode="after")
    def _validate_warmup_window(self) -> Settings:
        """Cross-field guard: MIN_SAMPLES must fit inside W.

        Without this, MIN_SAMPLES > W silently disables the z-score detector —
        the warm-up gate can never clear, so the rolling window is consulted
        but never trusted. Single-field validators don't catch the relationship,
        and a unit test that sets both knobs generously won't either. Fail fast
        at startup so the operator sees the bad config instead of a quiet
        dead detector.
        """
        if self.min_samples > self.w:
            raise ValueError(
                f"MIN_SAMPLES ({self.min_samples}) must be <= W ({self.w}); "
                "otherwise the warm-up gate never clears and the z-score "
                "detector silently never fires."
            )
        return self


settings = Settings()
