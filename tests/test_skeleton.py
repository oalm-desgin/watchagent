"""Milestone-1 smoke tests.

These exercise the imports and constants that the rest of the codebase will lean
on (cities tuple, settings load, logger factory). They are intentionally tiny —
the substantive tests live in test_dedup, test_detection, and test_api (M9).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from watchagent import __version__
from watchagent.cities import CITIES, CITY_BY_NAME
from watchagent.config import Settings
from watchagent.logging_setup import get_logger


def test_version_is_a_string() -> None:
    assert isinstance(__version__, str) and __version__


def test_three_cities_match_assignment() -> None:
    assert [c.name for c in CITIES] == ["Ottawa", "Toronto", "Vancouver"]
    assert CITY_BY_NAME["Ottawa"].latitude == 45.42
    assert CITY_BY_NAME["Toronto"].longitude == -79.42
    assert CITY_BY_NAME["Vancouver"].latitude == 49.25


def test_settings_have_safe_defaults() -> None:
    s = Settings()
    assert s.poll_interval_seconds > 0
    assert s.http_timeout_seconds > 0
    assert s.max_retries >= 0
    assert s.retry_backoff_base_seconds > 0
    assert s.w >= s.min_samples
    assert s.z_thresh > 0
    assert s.cooldown_seconds >= 0


def test_db_path_default_is_local_friendly() -> None:
    """Code default must work without a /data directory existing on the host."""
    s = Settings()
    assert not s.db_path.startswith("/data"), (
        "DB_PATH default must be a local path; the /data override belongs in "
        "docker-compose.yml so that pytest and ad-hoc local runs do not require "
        "the container's mount point to exist on the host."
    )


def test_min_samples_cannot_exceed_w() -> None:
    """Cross-field guard prevents the silent dead-detector bug.

    If MIN_SAMPLES > W the warm-up gate can never clear, so the z-score
    detector quietly stops firing forever. Per-field validators don't catch
    the relationship — only a model-level validator does.
    """
    with pytest.raises(ValidationError):
        Settings(w=4, min_samples=10)


def test_min_samples_equal_to_w_is_allowed() -> None:
    """The boundary case is valid: detector is permitted to require a full window."""
    s = Settings(w=10, min_samples=10)
    assert s.min_samples == s.w == 10


def test_logger_emits_without_raising() -> None:
    log = get_logger("tests.skeleton")
    log.info("smoke", note="logger initialized")
