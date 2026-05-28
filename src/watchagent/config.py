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

    # --- HTTP server -----------------------------------------------------
    # The bind address used by ``python -m watchagent``. ``0.0.0.0`` is
    # required for Docker port-mapping to reach the process; for a pure
    # local run, ``127.0.0.1`` is safer but the same default keeps
    # parity with the container.
    api_host: str = "0.0.0.0"
    api_port: int = Field(8000, ge=1, le=65535)

    # --- Test escape hatch -----------------------------------------------
    # Controls whether the lifespan launches the background poller. Always
    # True in production (and in docker-compose); set to False in unit
    # tests so the contract suite doesn't hit Open-Meteo on every run.
    # The poller is the only thing in the lifespan that makes outbound
    # HTTP calls; turning it off leaves the DB / engine / hydration path
    # intact for offline testing.
    enable_poller: bool = True

    # --- Detection: per-city temperature anomaly (z-score) ---------------
    w: int = Field(48, gt=1, description="rolling window size per city")
    min_samples: int = Field(6, ge=2, description="warm-up sample threshold")
    z_thresh: float = Field(2.5, gt=0)

    # Warm-up safety thresholds — used when the per-city window has fewer
    # than MIN_SAMPLES readings (cold start, fresh DB) and the z-score
    # baseline is statistically meaningless. These are deliberately
    # conservative absolute values for Canadian cities: anything ≥35 °C
    # or ≤−30 °C is notable regardless of any baseline.
    warmup_temp_high: float = Field(35.0, description="warm-up fallback °C high")
    warmup_temp_low: float = Field(-30.0, description="warm-up fallback °C low")

    # --- Detection: rapid rate-of-change ---------------------------------
    rate_thresh: float = Field(4.0, gt=0, description="°C / hour")

    # --- Detection: wind danger (absolute) -------------------------------
    wind_thresh: float = Field(40.0, gt=0, description="km/h sustained")

    # --- Detection: apparent-vs-actual divergence ------------------------
    divergence_thresh: float = Field(5.0, gt=0, description="°C")

    # --- Detection: heavy precipitation ----------------------------------
    # mm/h thresholds. moderate_thresh is the fire bar for the
    # heavy-precipitation detector; heavy_thresh raises severity to "high".
    # Onset (0 → >0) is its own detector with its own event type.
    precip_moderate_thresh: float = Field(4.0, gt=0, description="mm / hour")
    precip_heavy_thresh: float = Field(10.0, gt=0, description="mm / hour")

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

    @model_validator(mode="after")
    def _validate_precip_bands(self) -> Settings:
        """``precip_heavy_thresh`` should sit at or above ``precip_moderate_thresh``.

        If the bands are inverted, severity assignment is degenerate: every
        moderate fire would also be "high", which defeats the band entirely.
        """
        if self.precip_heavy_thresh < self.precip_moderate_thresh:
            raise ValueError(
                f"precip_heavy_thresh ({self.precip_heavy_thresh}) must be "
                f">= precip_moderate_thresh ({self.precip_moderate_thresh})"
            )
        return self


settings = Settings()
