"""Tests for the Debouncer — edge detection + cooldown with a frozen clock.

The injectable clock is what makes the cooldown-survives-restart test
deterministic. Every test here drives ``consume`` with explicit times; no
``sleep`` calls, no flakiness.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from watchagent.detection.cooldown import Debouncer


class FakeClock:
    """Tiny controllable clock for the debouncer.

    The Debouncer takes a ``Callable[[], datetime]``; this class is a
    deliberate stand-in for ``datetime.now(UTC)``. ``advance`` mutates the
    state-of-the-world; ``__call__`` is what the Debouncer reads."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock start must be timezone-aware")
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def advance(self, **kwargs: float) -> None:
        self._now = self._now + timedelta(**kwargs)


@pytest.fixture
def t0() -> datetime:
    return datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def clock(t0: datetime) -> FakeClock:
    return FakeClock(t0)


class TestConstructor:
    def test_negative_cooldown_rejected(self) -> None:
        with pytest.raises(ValueError):
            Debouncer(cooldown_seconds=-1)

    def test_zero_cooldown_allowed(self) -> None:
        """0 cooldown is "fire on every transition into anomalous, even
        immediately after another fire"; the spec doesn't forbid it."""
        d = Debouncer(cooldown_seconds=0)
        assert d is not None


class TestEdgeDetection:
    def test_first_anomalous_reading_fires(self, clock: FakeClock) -> None:
        """Cold start: no prior state, condition holds → fire."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        assert d.consume("Ottawa", "wind_danger", condition_holds=True) is True

    def test_clean_reading_does_not_fire(self, clock: FakeClock) -> None:
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        assert d.consume("Ottawa", "wind_danger", condition_holds=False) is False

    def test_clean_reading_arms_next_edge(self, clock: FakeClock) -> None:
        """The whole point of edge detection: clean → anomalous fires
        once, anomalous → clean re-arms, next anomalous fires again."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)

        d.consume("Ottawa", "wind", condition_holds=True)  # initial fire
        clock.advance(seconds=5)
        assert d.consume("Ottawa", "wind", condition_holds=False) is False
        clock.advance(seconds=5)
        # Cooldown not elapsed but condition cleared → re-arm; next True is an edge.
        assert d.consume("Ottawa", "wind", condition_holds=True) is True


class TestSustainedAnomalyAndCooldown:
    def test_sustained_anomaly_suppresses_until_cooldown_expires(
        self,
        clock: FakeClock,
    ) -> None:
        """The exact spec rule: suppress re-fire until the condition clears
        OR cooldown_seconds has elapsed. Here it never clears."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)

        assert d.consume("Ottawa", "wind", condition_holds=True) is True

        # Cumulative advances must stay strictly inside the 60-minute
        # cooldown — totals: 1, 11, 31, 56 minutes.
        for delta_minutes in (1, 10, 20, 25):
            clock.advance(minutes=delta_minutes)
            assert d.consume("Ottawa", "wind", condition_holds=True) is False

    def test_sustained_anomaly_refires_exactly_at_cooldown(
        self,
        clock: FakeClock,
    ) -> None:
        """At cooldown_seconds elapsed (inclusive), the next still-anomalous
        reading fires again. This is the "12-hour heatwave gets one event
        every 3h, not zero events after the first" property."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)

        assert d.consume("Ottawa", "wind", condition_holds=True) is True
        clock.advance(seconds=3600)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True

    def test_refire_resets_cooldown_anchor(self, clock: FakeClock) -> None:
        """After a re-fire on sustained anomaly, the next 3h are again
        suppressed — the cooldown anchor moves to the latest fire."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        d.consume("Ottawa", "wind", condition_holds=True)

        clock.advance(seconds=3600)
        d.consume("Ottawa", "wind", condition_holds=True)  # second fire

        clock.advance(minutes=30)  # 30 min after the second fire
        assert d.consume("Ottawa", "wind", condition_holds=True) is False


class TestKeyIsolation:
    def test_different_event_types_are_independent(self, clock: FakeClock) -> None:
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True
        assert d.consume("Ottawa", "heavy_precip", condition_holds=True) is True

    def test_different_cities_are_independent(self, clock: FakeClock) -> None:
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True
        assert d.consume("Toronto", "wind", condition_holds=True) is True


class TestHydration:
    def test_hydrate_within_cooldown_suppresses_first_fresh_anomaly(
        self,
        t0: datetime,
        clock: FakeClock,
    ) -> None:
        """The cooldown-survives-restart property: if we restart while a
        fire was inside its cooldown window, the next anomalous reading
        we see does NOT count as a fresh edge — it's a continuation, and
        the cooldown is what's already in effect."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        prior_fire = t0 - timedelta(seconds=600)  # 10 min ago, well inside 1h cooldown

        d.hydrate("Ottawa", "wind", last_fire_at=prior_fire)
        assert d.consume("Ottawa", "wind", condition_holds=True) is False

    def test_hydrate_outside_cooldown_allows_fresh_fire(
        self,
        t0: datetime,
        clock: FakeClock,
    ) -> None:
        """If the prior fire was OLDER than the cooldown window, treat it
        as 'no recent suppression' — the next anomalous reading fires.
        This is what should happen if the service was down for a day."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        prior_fire = t0 - timedelta(seconds=7200)  # 2h ago, outside 1h cooldown

        d.hydrate("Ottawa", "wind", last_fire_at=prior_fire)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True

    def test_hydrate_rejects_naive_datetime(self, clock: FakeClock) -> None:
        d = Debouncer(cooldown_seconds=3600, clock=clock)
        naive = datetime(2026, 5, 28, 12, 0, 0)
        with pytest.raises(ValueError):
            d.hydrate("Ottawa", "wind", last_fire_at=naive)


class TestObservedClearSemantics:
    """The subtle decision: an observed condition_holds=False is the OR's
    escape hatch in 'suppress until clears OR cooldown elapses'. So a
    genuine clear-then-re-enter within the cooldown DOES fire, but a
    continuously-held condition does NOT. Pin both directions explicitly
    so the behaviour can't drift in a refactor."""

    def test_observed_clear_then_re_enter_within_cooldown_fires(
        self,
        clock: FakeClock,
    ) -> None:
        """Sequence: True (fire) → False (clear) → True (within cooldown).

        The clear at step 2 takes the OR's escape hatch — suppression is
        over by the brief's reading. Step 3 is therefore a fresh edge
        and fires, even though only 10 minutes have passed since the
        original fire (well inside the 1h cooldown).
        """
        d = Debouncer(cooldown_seconds=3600, clock=clock)

        assert d.consume("Ottawa", "wind", condition_holds=True) is True

        clock.advance(minutes=5)
        assert d.consume("Ottawa", "wind", condition_holds=False) is False

        clock.advance(minutes=5)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True, (
            "An observed clear is the OR's escape hatch — the next "
            "True transition is a fresh edge and must fire even within "
            "the cooldown window."
        )

    def test_continuous_hold_stays_suppressed_until_cooldown(
        self,
        clock: FakeClock,
    ) -> None:
        """The other side of the same coin: without an observed clear,
        the cooldown is the only way out. 30 consecutive True readings
        over 30 minutes must produce exactly one fire."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)

        fires = 0
        for _ in range(30):
            if d.consume("Ottawa", "wind", condition_holds=True):
                fires += 1
            clock.advance(minutes=1)

        assert fires == 1, (
            "30 minutes of continuous anomaly inside a 60-minute "
            "cooldown must produce exactly one event."
        )

    def test_two_genuinely_separate_events_both_fire(
        self,
        clock: FakeClock,
    ) -> None:
        """The README sentence: an event that fires, fully clears, and
        re-occurs within the cooldown window IS two distinct events.
        We don't treat the second as a duplicate just because the wall
        clock hasn't ticked past the suppression window."""
        d = Debouncer(cooldown_seconds=3600, clock=clock)

        # Storm 1: blew through in 20 minutes.
        assert d.consume("Ottawa", "wind", condition_holds=True) is True
        clock.advance(minutes=20)
        assert d.consume("Ottawa", "wind", condition_holds=False) is False

        # Calm period.
        clock.advance(minutes=10)

        # Storm 2: a different storm, still within the original
        # storm's cooldown window. Must fire.
        assert d.consume("Ottawa", "wind", condition_holds=True) is True


class TestZeroCooldown:
    def test_zero_cooldown_refires_every_anomalous_reading(
        self,
        clock: FakeClock,
    ) -> None:
        """With cooldown_seconds=0 the only suppression layer is the
        in-anomalous edge flag. Once we're in_anomalous, the elapsed
        check (>= 0) is always true, so every subsequent anomalous
        reading re-fires."""
        d = Debouncer(cooldown_seconds=0, clock=clock)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True
        clock.advance(seconds=1)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True
        clock.advance(seconds=1)
        assert d.consume("Ottawa", "wind", condition_holds=True) is True
