"""Tests for CityState — the rolling-window invariant the whole module rests on."""

from __future__ import annotations

import pytest

from watchagent.detection.state import CityState
from watchagent.storage import Reading


def _r(city: str, t: str, temp: float) -> Reading:
    """Minimal Reading factory for state tests. Only the fields the state
    actually surfaces (city, reading_time_utc, temperature_2m) need to be
    realistic; the rest are arbitrary but valid."""
    return Reading(
        city=city,
        reading_time=t.replace("Z", ""),
        reading_time_utc=t,
        fetched_at="2026-05-28T00:00:00+00:00",
        temperature_2m=temp,
        apparent_temperature=temp,
        precipitation=0.0,
        wind_speed_10m=10.0,
        weather_code=0,
    )


class TestCapacityValidation:
    def test_capacity_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError):
            CityState(city="Ottawa", capacity=0)

    def test_capacity_one_is_valid(self) -> None:
        state = CityState(city="Ottawa", capacity=1)
        assert state.capacity == 1
        assert len(state) == 0


class TestWindowMechanics:
    def test_empty_state_has_no_last_and_no_temperatures(self) -> None:
        state = CityState(city="Ottawa", capacity=4)
        assert state.last is None
        assert state.temperatures() == []

    def test_add_appends_to_window(self) -> None:
        state = CityState(city="Ottawa", capacity=4)
        r = _r("Ottawa", "2026-05-28T10:00:00+00:00", 20.0)
        state.add(r)
        assert state.last is r
        assert state.temperatures() == [20.0]

    def test_window_evicts_oldest_at_capacity(self) -> None:
        """Capacity invariant — oldest is evicted FIFO so the window is
        always 'the most recent W priors'. This is what makes the
        z-score's baseline fresh, not stale."""
        state = CityState(city="Ottawa", capacity=3)
        for i, t in enumerate([10, 20, 30, 40, 50]):
            state.add(_r("Ottawa", f"2026-05-28T{10 + i:02d}:00:00+00:00", t))
        assert state.temperatures() == [30.0, 40.0, 50.0]

    def test_temperatures_preserves_chronological_order(self) -> None:
        state = CityState(city="Ottawa", capacity=4)
        for i, t in enumerate([10.0, 20.0, 30.0]):
            state.add(_r("Ottawa", f"2026-05-28T{10 + i:02d}:00:00+00:00", t))
        assert state.temperatures() == [10.0, 20.0, 30.0]


class TestHydrate:
    def test_hydrate_replaces_existing_window(self) -> None:
        state = CityState(city="Ottawa", capacity=4)
        state.add(_r("Ottawa", "2026-05-28T10:00:00+00:00", 99.0))
        state.hydrate(
            [
                _r("Ottawa", "2026-05-27T10:00:00+00:00", 1.0),
                _r("Ottawa", "2026-05-27T11:00:00+00:00", 2.0),
            ]
        )
        assert state.temperatures() == [1.0, 2.0]

    def test_hydrate_respects_capacity(self) -> None:
        """If we hand in more readings than capacity allows, only the last
        ``capacity`` survive — same behaviour as ``add``."""
        state = CityState(city="Ottawa", capacity=2)
        state.hydrate(
            [
                _r("Ottawa", "2026-05-27T10:00:00+00:00", 1.0),
                _r("Ottawa", "2026-05-27T11:00:00+00:00", 2.0),
                _r("Ottawa", "2026-05-27T12:00:00+00:00", 3.0),
            ]
        )
        assert state.temperatures() == [2.0, 3.0]


class TestDisjointBaselineInvariant:
    """The single most important behavioural test: when ``evaluate`` runs,
    the window does NOT contain the reading under evaluation. This is
    asserted at the engine boundary in test_detection_engine.py; here we
    pin the lower-level invariant that ``add`` is what introduces the new
    point, and ``temperatures()`` mid-evaluation reflects only priors."""

    def test_temperatures_reflects_only_priors_until_add_called(self) -> None:
        state = CityState(city="Ottawa", capacity=10)
        state.add(_r("Ottawa", "2026-05-28T10:00:00+00:00", 20.0))
        state.add(_r("Ottawa", "2026-05-28T11:00:00+00:00", 21.0))

        new_reading = _r("Ottawa", "2026-05-28T12:00:00+00:00", 35.0)

        assert state.temperatures() == [20.0, 21.0]
        assert 35.0 not in state.temperatures()

        state.add(new_reading)
        assert state.temperatures() == [20.0, 21.0, 35.0]
