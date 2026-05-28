"""Tests for the WMO weather code registry.

The most important assertion in this file is ``test_tiers_are_strictly_ordered``:
detector #5's "categorical jump" semantics depend on it. If that ever breaks,
every weather-code-transition event is suspect.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from watchagent import weather_codes
from watchagent.weather_codes import (
    OPEN_METEO_PUBLISHED_CODES,
    WMO_CODES,
    Severity,
    Tier,
    description_of,
    is_more_severe,
    lookup,
    severity_of,
    tier_of,
)

# ---------------------------------------------------------------------------
# Tier ordering — load-bearing for detector #5.
# ---------------------------------------------------------------------------


def test_tiers_are_strictly_ordered() -> None:
    """Detector #5 relies on this ordering for `is this categorically worse?`."""
    assert (
        Tier.CLEAR
        < Tier.CLOUDY
        < Tier.FOG
        < Tier.DRIZZLE
        < Tier.RAIN
        < Tier.SNOW
        < Tier.FREEZING_PRECIPITATION
        < Tier.THUNDERSTORM
    )


def test_freezing_precipitation_sits_above_snow() -> None:
    """Domain decision: ice accretion is more disruptive than equivalent snowfall.

    Documented in the module docstring (1998 Ottawa ice storm context).
    """
    assert Tier.FREEZING_PRECIPITATION > Tier.SNOW
    assert Tier.FREEZING_PRECIPITATION > Tier.RAIN


# ---------------------------------------------------------------------------
# Registry completeness — pinned against Open-Meteo's published set.
# ---------------------------------------------------------------------------


def test_registry_covers_open_meteo_set() -> None:
    """Every code Open-Meteo emits MUST be in the registry — silent gaps are
    invisible dead spots in detector #5 because unknown codes return False
    from is_more_severe by design. Pin the set so a refactor can't drop one.
    """
    assert set(WMO_CODES.keys()) == set(OPEN_METEO_PUBLISHED_CODES)


def test_no_duplicate_codes_in_registry() -> None:
    """Each WMO code should appear exactly once in the registry."""
    codes = [rec.code for rec in WMO_CODES.values()]
    assert len(codes) == len(set(codes))


def test_registry_keys_match_record_codes() -> None:
    """Catch a future refactor that desyncs the dict key from the record's code field."""
    for key, rec in WMO_CODES.items():
        assert key == rec.code


def test_every_known_code_round_trips() -> None:
    for code, rec in WMO_CODES.items():
        weather_codes._clear_unknown_codes_cache()
        assert lookup(code) is rec
        assert tier_of(code) is rec.tier
        assert severity_of(code) is rec.severity
        assert description_of(code) == rec.description


# ---------------------------------------------------------------------------
# Bucket assignments — spot-check the consequential codes.
# ---------------------------------------------------------------------------


def test_canonical_buckets() -> None:
    """Spot-check the consequential codes land in the expected tier."""
    assert tier_of(0) is Tier.CLEAR
    assert tier_of(3) is Tier.CLOUDY
    assert tier_of(45) is Tier.FOG
    assert tier_of(55) is Tier.DRIZZLE
    assert tier_of(65) is Tier.RAIN
    assert tier_of(75) is Tier.SNOW
    assert tier_of(95) is Tier.THUNDERSTORM
    assert tier_of(99) is Tier.THUNDERSTORM


@pytest.mark.parametrize("code", [56, 57, 66, 67])
def test_freezing_codes_live_in_freezing_tier(code: int) -> None:
    """All four freezing-precip codes belong to the elevated tier (above SNOW)."""
    assert tier_of(code) is Tier.FREEZING_PRECIPITATION


@pytest.mark.parametrize("code", [56, 57, 66, 67])
def test_freezing_codes_are_high_severity(code: int) -> None:
    """Even 'light' freezing precip is HIGH severity — ice accretion threshold
    is low and the disruption is sustained (power lines, road glaze).
    """
    assert severity_of(code) is Severity.HIGH


def test_severity_values_are_the_expected_strings() -> None:
    """Severity is stored as TEXT on the event row; the values must be stable."""
    assert Severity.LOW == "low"
    assert Severity.MEDIUM == "medium"
    assert Severity.HIGH == "high"


# ---------------------------------------------------------------------------
# Unknown codes — fail-safe (silent) AND observable (one-shot log).
# ---------------------------------------------------------------------------


def test_unknown_code_returns_none() -> None:
    weather_codes._clear_unknown_codes_cache()
    assert lookup(999) is None
    assert tier_of(998) is None
    assert severity_of(997) is None


def test_description_of_falls_back_for_unknown() -> None:
    """Logs and event reasons stay readable even if the WMO table updates."""
    weather_codes._clear_unknown_codes_cache()
    assert description_of(999) == "Unknown WMO code 999"


def test_unknown_code_logs_once_then_dedupes() -> None:
    """First sighting of an unknown code emits weather_code.unknown at INFO;
    subsequent sightings of the same code are silent."""
    weather_codes._clear_unknown_codes_cache()
    with capture_logs() as cap:
        lookup(999)
        lookup(999)
        lookup(999)
        tier_of(998)
        severity_of(998)
        description_of(997)
        is_more_severe(996, 0)

    unknown_logs = [c for c in cap if c.get("event") == "weather_code.unknown"]
    seen = {c["weather_code"] for c in unknown_logs}
    assert seen == {999, 998, 997, 996}, (
        "Each distinct unknown code should log exactly once"
    )
    assert len(unknown_logs) == 4, (
        f"Expected 4 logs (one per distinct unknown code), got {len(unknown_logs)}"
    )


def test_unknown_code_log_carries_useful_fields() -> None:
    """The log line must carry the code so operators can find it in stored data."""
    weather_codes._clear_unknown_codes_cache()
    with capture_logs() as cap:
        lookup(12345)
    target = next(c for c in cap if c.get("event") == "weather_code.unknown")
    assert target["weather_code"] == 12345
    assert target["log_level"] == "info"
    assert "tier-transition events suppressed" in target["note"]


# ---------------------------------------------------------------------------
# is_more_severe — detector #5's firing condition.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("new_code", "old_code", "expected"),
    [
        # Cross-tier upgrade: detector #5 fires.
        (95, 0, True),    # CLEAR -> THUNDERSTORM
        (75, 0, True),    # CLEAR -> SNOW
        (65, 1, True),    # CLOUDY -> RAIN
        (45, 0, True),    # CLEAR -> FOG
        (3, 0, True),     # CLEAR -> CLOUDY
        # Freezing precipitation sits above plain rain and snow.
        (66, 65, True),   # RAIN heavy -> FREEZING_PRECIPITATION
        (56, 75, True),   # SNOW heavy -> FREEZING_PRECIPITATION light freezing drizzle
        (95, 67, True),   # FREEZING_PRECIPITATION -> THUNDERSTORM (top tier)
        # Same tier (no jump) — within-tier intensity is detector #4's job.
        (61, 63, False),  # RAIN slight  -> RAIN moderate
        (1, 3, False),    # CLOUDY mainly clear -> CLOUDY overcast
        (95, 99, False),  # THUNDERSTORM -> THUNDERSTORM (heavier hail)
        (51, 55, False),  # DRIZZLE light -> DRIZZLE dense
        (66, 67, False),  # FREEZING_PRECIPITATION light -> heavy (within tier)
        (56, 57, False),  # FREEZING drizzle light -> dense (within tier)
        # Downgrades are never "more severe".
        (0, 95, False),
        (3, 65, False),
        (75, 95, False),  # THUNDERSTORM -> SNOW is a downgrade for new_code=75
        (95, 75, True),   # SNOW -> THUNDERSTORM upgrade
        (65, 66, False),  # FREEZING_PRECIPITATION -> RAIN downgrade
        # Unknown codes: silence rather than false fire.
        (999, 0, False),
        (0, 999, False),
        (999, 998, False),
    ],
)
def test_is_more_severe_tier_transitions(
    new_code: int, old_code: int, expected: bool
) -> None:
    weather_codes._clear_unknown_codes_cache()
    assert is_more_severe(new_code, old_code) is expected
