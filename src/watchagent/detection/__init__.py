"""Detection module — turns raw readings into notable events.

Three layers, deliberately separated:

* :mod:`.state`       — per-city rolling window of *prior* readings.
* :mod:`.detectors`   — pure functions of (current reading, prior state) that
                        say "does the condition hold *for this reading*?". No
                        time, no cooldown, no I/O.
* :mod:`.cooldown`    — a debouncer with an injectable clock that decides
                        whether a candidate actually fires given the last
                        emission time per ``(city, event_type)``.
* :mod:`.engine`      — orchestrator: hydrates state from the DB, drives the
                        detectors per reading, persists fired events.

This split is the point of the whole milestone:

* "did the condition hold?" is a unit-testable pure function;
* "should we emit an event?" is a unit-testable cooldown layer with a frozen
  clock — so the cooldown-survives-restart test is deterministic.

Hydration is strict: per-city windows from
:func:`Database.recent_readings_for_city` and per-(city, event_type) cooldowns
from :func:`Database.latest_event` MUST both complete before the first poll
cycle. The lifespan in M7 awaits :meth:`DetectionEngine.hydrate_from_db` and
only then schedules the poller.
"""

from __future__ import annotations

from .cooldown import Debouncer
from .detectors import (
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
from .engine import DetectionEngine, build_default_detectors
from .state import CityState

__all__ = [
    "CandidateEvent",
    "CityState",
    "Debouncer",
    "DetectionEngine",
    "Detector",
    "FeelsLikeDivergenceDetector",
    "HeavyPrecipitationDetector",
    "PrecipitationOnsetDetector",
    "RapidTempChangeDetector",
    "TemperatureAnomalyDetector",
    "WeatherCodeTransitionDetector",
    "WindDangerDetector",
    "build_default_detectors",
]
