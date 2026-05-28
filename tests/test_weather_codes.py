"""Tests for the WMO weather code registry.

The most important assertion in this file is ``test_tiers_are_strictly_ordered``:
detector #5's "categorical jump" semantics depend on it. If that ever breaks,
every weather-code-transition event is suspect.
"""

from __future__ import annotations

import pytest

from watchagent.weather_codes import (
    WMO_CODES,
    Severity,
    Tier,
    description_of,
    is_more_severe,
    lookup,
    severity_of,
    tier_of,
)


def test_tiers_are_strictly_ordered() -> None:
    """Detector #5 relies on this ordering for `is this categorically worse?`."""
    assert (
        Tier.CLEAR
        < Tier.CLOUDY
        < Tier.FOG
        < Tier.DRIZZLE
        < Tier.RAIN
        < Tier.SNOW
        < Tier.THUNDERSTORM
    )


def test_every_known_code_round_trips() -> None:
    for code, rec in WMO_CODES.items():
        assert lookup(code) is rec
        assert tier_of(code) is rec.tier
        assert severity_of(code) is rec.severity
        assert description_of(code) == rec.description


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


def test_severity_values_are_the_expected_strings() -> None:
    """Severity is stored as TEXT on the event row; the values must be stable."""
    assert Severity.LOW == "low"
    assert Severity.MEDIUM == "medium"
    assert Severity.HIGH == "high"


def test_unknown_code_returns_none() -> None:
    assert lookup(999) is None
    assert tier_of(999) is None
    assert severity_of(999) is None


def test_description_of_falls_back_for_unknown() -> None:
    """Logs and event reasons stay readable even if the WMO table updates."""
    assert description_of(999) == "Unknown WMO code 999"


@pytest.mark.parametrize(
    ("new_code", "old_code", "expected"),
    [
        # Cross-tier upgrade: detector #5 fires.
        (95, 0, True),    # CLEAR -> THUNDERSTORM
        (75, 0, True),    # CLEAR -> SNOW
        (65, 1, True),    # CLOUDY -> RAIN
        (45, 0, True),    # CLEAR -> FOG
        (3, 0, True),     # CLEAR -> CLOUDY
        # Same tier (no jump) — within-tier intensity is detector #4's job.
        (61, 63, False),  # RAIN slight  -> RAIN moderate
        (1, 3, False),    # CLOUDY mainly clear -> CLOUDY overcast
        (95, 99, False),  # THUNDERSTORM -> THUNDERSTORM (heavier hail)
        (51, 55, False),  # DRIZZLE light -> DRIZZLE dense
        # Downgrades are never "more severe".
        (0, 95, False),
        (3, 65, False),
        (75, 95, False),  # SNOW <- THUNDERSTORM is a downgrade for new_code=75
        (95, 75, True),   # SNOW -> THUNDERSTORM (sanity: still a real upgrade)
        # Unknown codes: silence rather than false fire.
        (999, 0, False),
        (0, 999, False),
        (999, 998, False),
    ],
)
def test_is_more_severe_tier_transitions(
    new_code: int, old_code: int, expected: bool
) -> None:
    assert is_more_severe(new_code, old_code) is expected


def test_no_duplicate_codes_in_registry() -> None:
    """Each WMO code should appear exactly once in the registry."""
    codes = [rec.code for rec in WMO_CODES.values()]
    assert len(codes) == len(set(codes))


def test_registry_keys_match_record_codes() -> None:
    """Catch a future refactor that desyncs the dict key from the record's code field."""
    for key, rec in WMO_CODES.items():
        assert key == rec.code
