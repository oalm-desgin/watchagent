"""Engine integration tests.

These run against a real ``Database`` (on-disk SQLite via the existing
M3 fixture) so we exercise hydration end-to-end. The tests pin the four
behaviours the milestone hinges on:

1. The disjoint-baseline contract holds at the engine level: the new
   reading is added AFTER all detectors run, so each detector sees only
   priors.
2. Hydration replays prior readings into per-city windows AND seeds the
   debouncer's ``last_fire_at`` from the events table — no duplicate
   re-fires across restarts.
3. A bug inside one detector does not poison the rest of the cycle or
   the engine state.
4. ``on_new_reading`` returns the events that fired, in order, and they
   are persisted to the DB with the correct shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from watchagent.cities import CITIES, City
from watchagent.detection.cooldown import Debouncer
from watchagent.detection.detectors import (
    CandidateEvent,
    Detector,
    HeavyPrecipitationDetector,
    PrecipitationOnsetDetector,
    TemperatureAnomalyDetector,
    WindDangerDetector,
)
from watchagent.detection.engine import DetectionEngine
from watchagent.detection.state import CityState
from watchagent.storage import Database, Event, Reading

OTTAWA, TORONTO, VANCOUVER = CITIES

# Re-export City so type hints in helpers below are explicit; the symbol is
# imported above purely for that purpose.
_ = City


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(path=str(tmp_path / "engine.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


def make_reading(
    *,
    city: str = "Ottawa",
    when: str = "2026-05-28T12:00:00+00:00",
    temperature: float = 20.0,
    apparent: float | None = None,
    precipitation: float = 0.0,
    wind: float = 10.0,
    weather_code: int = 0,
) -> Reading:
    return Reading(
        city=city,
        reading_time=when.split("+", 1)[0],
        reading_time_utc=when,
        fetched_at="2026-05-28T12:00:30+00:00",
        temperature_2m=temperature,
        apparent_temperature=apparent if apparent is not None else temperature,
        precipitation=precipitation,
        wind_speed_10m=wind,
        weather_code=weather_code,
    )


# ---------------------------------------------------------------------------
# Disjoint baseline (the headline invariant)
# ---------------------------------------------------------------------------


class TestDisjointBaselineInvariant:
    @pytest.mark.asyncio
    async def test_z_score_does_not_see_current_reading_in_baseline(
        self,
        db: Database,
    ) -> None:
        """Drive the engine with a city that has 20 priors at exactly 22°C.
        The current reading is 35°C. The z-score detector should see
        priors=[22.0]*20 (std=0 → skip). If the engine accidentally added
        the current reading FIRST, the baseline would have one 35.0 and
        twenty 22.0s — std would be non-zero and z would be ~3, and the
        detector would fire. Pin that this does NOT happen."""
        z_detector = TemperatureAnomalyDetector(
            min_samples=6, z_thresh=2.5, warmup_high=999.0, warmup_low=-999.0
        )
        debouncer = Debouncer(cooldown_seconds=3600)
        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[z_detector],
            debouncer=debouncer,
            window_capacity=24,
        )

        for i in range(20):
            engine.state_for("Ottawa").add(
                make_reading(when=f"2026-05-28T{(i % 24):02d}:00:00+00:00", temperature=22.0)
            )

        events = await engine.on_new_reading(
            make_reading(when="2026-05-29T12:00:00+00:00", temperature=35.0)
        )

        assert events == [], (
            "z-score must NOT fire when the prior window is flat — std=0 → "
            "baseline is degenerate. If this fires, the current reading is "
            "leaking into the baseline."
        )

        assert engine.state_for("Ottawa").temperatures()[-1] == 35.0


# ---------------------------------------------------------------------------
# Per-detector failure isolation
# ---------------------------------------------------------------------------


class _BrokenDetector:
    event_type = "broken"

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        raise RuntimeError("kaboom")


class _AlwaysFiringDetector:
    event_type = "always"

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        return CandidateEvent(
            event_type=self.event_type,
            severity="low",
            reason="always fires",
            context={"why": "test"},
        )


class TestDetectorFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_detector_crashing_does_not_break_the_others(
        self,
        db: Database,
    ) -> None:
        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[_BrokenDetector(), _AlwaysFiringDetector()],
            debouncer=Debouncer(cooldown_seconds=3600),
            window_capacity=4,
        )
        events = await engine.on_new_reading(make_reading())
        assert len(events) == 1
        assert events[0].event_type == "always"

    @pytest.mark.asyncio
    async def test_crashing_detector_clears_in_anomalous_flag(
        self,
        db: Database,
    ) -> None:
        """The crashing branch must update the debouncer too — otherwise
        a once-anomalous flag could get stuck and silently suppress
        future fires."""
        debouncer = Debouncer(cooldown_seconds=3600)
        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[_BrokenDetector()],
            debouncer=debouncer,
            window_capacity=4,
        )
        await engine.on_new_reading(make_reading())
        # in_anomalous should be False (set explicitly by the engine's
        # exception handler), so a future condition_holds=True would be
        # treated as a fresh edge.
        assert debouncer.state_for("Ottawa", "broken").in_anomalous is False


# ---------------------------------------------------------------------------
# Persistence + return shape
# ---------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_fired_event_is_persisted_with_full_context(
        self,
        db: Database,
    ) -> None:
        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[WindDangerDetector(threshold=40.0)],
            debouncer=Debouncer(cooldown_seconds=3600),
            window_capacity=4,
        )
        events = await engine.on_new_reading(make_reading(wind=85.0))
        assert len(events) == 1
        ev = events[0]
        assert ev.id is not None  # DB assigned an ID
        assert ev.event_type == "wind_danger"
        assert ev.severity == "high"
        assert ev.context["wind_kmh"] == 85.0
        assert "85.0" in ev.reason

        rows = await db.select_events(city="Ottawa")
        assert len(rows) == 1
        assert rows[0].context == ev.context

    @pytest.mark.asyncio
    async def test_unknown_city_is_logged_and_skipped(self, db: Database) -> None:
        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[WindDangerDetector(threshold=40.0)],
            debouncer=Debouncer(cooldown_seconds=3600),
            window_capacity=4,
        )
        events = await engine.on_new_reading(
            make_reading(city="Atlantis", wind=85.0)
        )
        assert events == []


# ---------------------------------------------------------------------------
# Multi-detector wiring + cooldown across calls
# ---------------------------------------------------------------------------


class TestMultiDetectorEngine:
    @pytest.mark.asyncio
    async def test_independent_detector_cooldowns(self, db: Database) -> None:
        """Wind firing should not affect heavy-precip's cooldown."""
        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[
                WindDangerDetector(threshold=40.0),
                HeavyPrecipitationDetector(moderate_thresh=4.0, heavy_thresh=10.0),
            ],
            debouncer=Debouncer(cooldown_seconds=3600),
            window_capacity=4,
        )

        # First reading: wind only.
        events = await engine.on_new_reading(
            make_reading(when="2026-05-28T10:00:00+00:00", wind=85.0)
        )
        assert {e.event_type for e in events} == {"wind_danger"}

        # Second reading: still windy + heavy precip starts.
        events = await engine.on_new_reading(
            make_reading(when="2026-05-28T11:00:00+00:00", wind=85.0, precipitation=12.0)
        )
        # Wind suppressed (in cooldown); heavy_precip fires fresh.
        assert {e.event_type for e in events} == {"heavy_precipitation"}

    @pytest.mark.asyncio
    async def test_onset_then_clear_then_onset_fires_twice(
        self,
        db: Database,
    ) -> None:
        """Edge re-arming: after a fire on transition, a clean reading
        clears in_anomalous, and the next anomalous reading fires again.
        With cooldown=0 the only suppression is the edge flag — so this
        exercises the edge-detection layer specifically."""
        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[PrecipitationOnsetDetector()],
            debouncer=Debouncer(cooldown_seconds=0),
            window_capacity=4,
        )

        await engine.on_new_reading(
            make_reading(when="2026-05-28T10:00:00+00:00", precipitation=0.0)
        )
        e1 = await engine.on_new_reading(
            make_reading(when="2026-05-28T11:00:00+00:00", precipitation=0.5)
        )
        assert len(e1) == 1  # onset fires

        await engine.on_new_reading(
            make_reading(when="2026-05-28T12:00:00+00:00", precipitation=0.0)
        )  # condition cleared, no fire
        e3 = await engine.on_new_reading(
            make_reading(when="2026-05-28T13:00:00+00:00", precipitation=0.7)
        )
        assert len(e3) == 1  # second onset fires too


# ---------------------------------------------------------------------------
# Hydration end-to-end
# ---------------------------------------------------------------------------


def _all_three_cities() -> tuple[City, ...]:
    return (OTTAWA, TORONTO, VANCOUVER)


class TestHydration:
    @pytest.mark.asyncio
    async def test_hydrate_loads_priors_into_per_city_windows(
        self,
        db: Database,
    ) -> None:
        for i, t in enumerate([15.0, 16.0, 17.0]):
            await db.insert_reading(
                make_reading(
                    when=f"2026-05-28T{10 + i:02d}:00:00+00:00",
                    temperature=t,
                )
            )
        for i, t in enumerate([20.0, 21.0]):
            await db.insert_reading(
                make_reading(
                    city="Toronto",
                    when=f"2026-05-28T{10 + i:02d}:00:00-04:00",
                    temperature=t,
                )
            )

        engine = DetectionEngine(
            db=db,
            cities=_all_three_cities(),
            detectors=build_default_detectors_with_test_settings(),
            debouncer=Debouncer(cooldown_seconds=3600),
            window_capacity=10,
        )
        await engine.hydrate_from_db()

        assert engine.hydrated is True
        assert engine.state_for("Ottawa").temperatures() == [15.0, 16.0, 17.0]
        assert engine.state_for("Toronto").temperatures() == [20.0, 21.0]
        assert engine.state_for("Vancouver").temperatures() == []

    @pytest.mark.asyncio
    async def test_hydrate_seeds_cooldown_from_events(self, db: Database) -> None:
        """Cooldown-survives-restart: insert a fired event into the DB,
        construct a fresh engine, and verify the next anomalous reading
        does NOT re-fire (because hydration set in_anomalous=True with
        the original last_fire_at)."""
        # Seed a fire 10 minutes before "now"
        now = datetime.now(UTC)
        recent_iso = now.replace(microsecond=0).isoformat()
        await db.insert_event(
            Event(
                id=None,
                city="Ottawa",
                event_type="wind_danger",
                reading_time="2026-05-28T11:00:00",
                reading_time_utc="2026-05-28T15:00:00+00:00",
                detected_at=recent_iso,
                severity="high",
                reason="seeded",
                context={"wind_kmh": 90.0, "threshold_kmh": 40.0},
            )
        )

        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[WindDangerDetector(threshold=40.0)],
            debouncer=Debouncer(cooldown_seconds=3600),
            window_capacity=4,
        )
        await engine.hydrate_from_db()

        # First anomalous reading after restart — should be SUPPRESSED
        # because the previous fire is inside the 1h cooldown window.
        events = await engine.on_new_reading(
            make_reading(when="2026-05-28T15:30:00+00:00", wind=85.0)
        )
        assert events == [], (
            "Hydration must seed the debouncer so a fresh process does not "
            "re-emit an event whose cooldown is still in effect."
        )

    @pytest.mark.asyncio
    async def test_hydrate_with_old_event_allows_fresh_fire(
        self,
        db: Database,
    ) -> None:
        """If the DB has an event but it's well outside the cooldown
        window, the next anomalous reading SHOULD fire. The hydration
        logic must not silently suppress legitimate new events."""
        await db.insert_event(
            Event(
                id=None,
                city="Ottawa",
                event_type="wind_danger",
                reading_time="2025-01-01T10:00:00",
                reading_time_utc="2025-01-01T15:00:00+00:00",
                detected_at="2025-01-01T15:00:00+00:00",
                severity="high",
                reason="ancient",
                context={"wind_kmh": 90.0, "threshold_kmh": 40.0},
            )
        )

        engine = DetectionEngine(
            db=db,
            cities=(OTTAWA,),
            detectors=[WindDangerDetector(threshold=40.0)],
            debouncer=Debouncer(cooldown_seconds=3600),
            window_capacity=4,
        )
        await engine.hydrate_from_db()

        events = await engine.on_new_reading(
            make_reading(wind=85.0)
        )
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_default_detectors_with_test_settings() -> list[Detector]:
    """Same shape as :func:`build_default_detectors` but with explicit
    knobs so this file does not need to construct a full Settings."""
    return [
        TemperatureAnomalyDetector(
            min_samples=6, z_thresh=2.5, warmup_high=35.0, warmup_low=-30.0
        ),
        WindDangerDetector(threshold=40.0),
        HeavyPrecipitationDetector(moderate_thresh=4.0, heavy_thresh=10.0),
        PrecipitationOnsetDetector(),
    ]
