"""Detection engine — wires detectors, debouncer, and storage.

Responsibilities, in order:

1. Hold per-city :class:`CityState` windows (one per known city).
2. Hydrate those windows + the cooldown layer from the DB on startup.
3. For each new reading from the poller, run every detector against the
   pre-add state, ask the debouncer per-detector whether to fire, persist
   any firing events, then add the reading to the city's window.
4. Return the list of events that fired, both for log visibility and so
   M9 / API tests can assert on the integration end-to-end.

The engine is pure async + dependency-injected. M7 constructs it inside
the FastAPI lifespan and hands it to the poller as the
``on_new_reading`` callback. Strict ordering, as promised: hydration
finishes before the first poll cycle starts.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from watchagent.detection.cooldown import Debouncer
from watchagent.detection.detectors import (
    CandidateEvent,
    Detector,
    FeelsLikeDivergenceDetector,
    HeavyPrecipitationDetector,
    PrecipitationOnsetDetector,
    RapidTempChangeDetector,
    TemperatureAnomalyDetector,
    WeatherCodeTransitionDetector,
    WindDangerDetector,
)
from watchagent.detection.state import CityState
from watchagent.storage import Event, Reading, utc_now_iso

if TYPE_CHECKING:
    from collections.abc import Sequence

    from watchagent.cities import City
    from watchagent.config import Settings
    from watchagent.storage import Database


log = structlog.get_logger(__name__)


def build_default_detectors(settings: Settings) -> list[Detector]:
    """Construct the full detector roster from ``Settings``.

    Kept as a free function (rather than baked into the engine) so tests
    can wire a single detector or a stub without going through Settings.
    """
    return [
        TemperatureAnomalyDetector(
            min_samples=settings.min_samples,
            z_thresh=settings.z_thresh,
            warmup_high=settings.warmup_temp_high,
            warmup_low=settings.warmup_temp_low,
        ),
        RapidTempChangeDetector(rate_thresh=settings.rate_thresh),
        WindDangerDetector(threshold=settings.wind_thresh),
        PrecipitationOnsetDetector(),
        HeavyPrecipitationDetector(
            moderate_thresh=settings.precip_moderate_thresh,
            heavy_thresh=settings.precip_heavy_thresh,
        ),
        WeatherCodeTransitionDetector(),
        FeelsLikeDivergenceDetector(threshold=settings.divergence_thresh),
    ]


class DetectionEngine:
    """Runs detectors per reading, debounces candidates, persists events."""

    def __init__(
        self,
        *,
        db: Database,
        cities: Sequence[City],
        detectors: Sequence[Detector],
        debouncer: Debouncer,
        window_capacity: int,
    ) -> None:
        if window_capacity < 1:
            raise ValueError("window_capacity must be >= 1")
        self._db = db
        self._cities = tuple(cities)
        self._detectors = tuple(detectors)
        self._debouncer = debouncer
        self._states: dict[str, CityState] = {
            c.name: CityState(city=c.name, capacity=window_capacity)
            for c in self._cities
        }
        self._hydrated = False

    # -- properties (read-only) ------------------------------------------

    @property
    def cities(self) -> tuple[City, ...]:
        return self._cities

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return self._detectors

    @property
    def hydrated(self) -> bool:
        return self._hydrated

    def state_for(self, city: str) -> CityState:
        """Read-only access to a city's state (used by tests)."""
        return self._states[city]

    # -- hydration -------------------------------------------------------

    async def hydrate_from_db(self) -> None:
        """Reload per-city windows + cooldown state from the DB.

        This MUST be awaited before the first poll cycle starts. Skipping
        it would mean the first reading after restart re-fires events
        that were already emitted in the previous run, and the z-score
        starts from an empty window even when 47 prior readings are sitting
        in the database. The lifespan in M7 awaits this inside startup
        and only then schedules the poller task.
        """
        log.info("detection.hydrate.start")

        for city in self._cities:
            readings = await self._db.recent_readings_for_city(
                city.name,
                limit=self._states[city.name].capacity,
            )
            self._states[city.name].hydrate(readings)
            log.info(
                "detection.hydrate.window",
                city=city.name,
                readings_loaded=len(readings),
                window_capacity=self._states[city.name].capacity,
            )

        for city in self._cities:
            for detector in self._detectors:
                latest = await self._db.latest_event(city.name, detector.event_type)
                if latest is None:
                    continue
                try:
                    last_fire_at = datetime.fromisoformat(latest.detected_at)
                except ValueError:
                    log.warning(
                        "detection.hydrate.bad_detected_at",
                        city=city.name,
                        event_type=detector.event_type,
                        detected_at=latest.detected_at,
                    )
                    continue
                self._debouncer.hydrate(
                    city.name,
                    detector.event_type,
                    last_fire_at=last_fire_at,
                )

        self._hydrated = True
        log.info("detection.hydrate.done")

    # -- main entry point (poller hands us new readings here) ------------

    async def on_new_reading(self, reading: Reading) -> list[Event]:
        """Run all detectors for ``reading``, persist any fires, return events.

        Order of operations:

        1. Snapshot the current city's state (the window holds *priors*).
        2. For each detector, ``evaluate(reading, state)`` → CandidateEvent | None.
        3. Pass every (city, event_type) outcome to the debouncer — including
           the ``None`` ones, so the in-anomalous flag clears when a condition
           ceases to hold.
        4. For approved candidates, build a real ``Event`` and insert.
        5. Add the reading to the window — AFTER all detectors have run.

        Step 5 is what guarantees "the window the z-score sees and the
        point it judges are disjoint".
        """
        state = self._states.get(reading.city)
        if state is None:
            # An unknown city showed up in the poller stream — log and
            # skip rather than raise. Cities are a fixed contract; if
            # this fires, something upstream lost its mind.
            log.warning("detection.unknown_city", city=reading.city)
            return []

        fired: list[Event] = []

        for detector in self._detectors:
            try:
                candidate = detector.evaluate(reading, state)
            except Exception:
                # Detector bug → log with full context, skip this detector
                # for this reading, KEEP the rest of the cycle running.
                log.exception(
                    "detection.detector.error",
                    city=reading.city,
                    event_type=detector.event_type,
                    reading_time_utc=reading.reading_time_utc,
                )
                # Treat as "condition didn't hold" so the in-anomalous
                # flag clears if it was set, rather than getting stuck.
                self._debouncer.consume(
                    reading.city,
                    detector.event_type,
                    condition_holds=False,
                )
                continue

            should_fire = self._debouncer.consume(
                reading.city,
                detector.event_type,
                condition_holds=candidate is not None,
            )
            if not should_fire or candidate is None:
                continue

            event = await self._persist(reading, candidate)
            fired.append(event)

        state.add(reading)
        return fired

    # -- internals -------------------------------------------------------

    async def _persist(self, reading: Reading, candidate: CandidateEvent) -> Event:
        event = Event(
            id=None,
            city=reading.city,
            event_type=candidate.event_type,
            reading_time=reading.reading_time,
            reading_time_utc=reading.reading_time_utc,
            detected_at=utc_now_iso(),
            severity=candidate.severity,
            reason=candidate.reason,
            context=candidate.context,
        )
        stored = await self._db.insert_event(event)
        log.info(
            "detection.event.fired",
            city=stored.city,
            event_type=stored.event_type,
            severity=stored.severity,
            reading_time_utc=stored.reading_time_utc,
            detected_at=stored.detected_at,
            reason=stored.reason,
            # Compact JSON of context keeps the structured log searchable
            # without flattening every numeric into a top-level field.
            context_json=json.dumps(stored.context, sort_keys=True),
        )
        return stored
