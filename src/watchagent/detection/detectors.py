"""Notable-event detectors.

Each detector is a callable class that exposes:

* ``event_type: str`` — the stable identifier persisted to the DB and
  surfaced in the API. One detector ↔ one event type.
* ``evaluate(reading, state) -> CandidateEvent | None`` — pure function. No
  time, no cooldown, no I/O. Returns ``None`` if the condition does not
  hold for this reading; otherwise a :class:`CandidateEvent` carrying both
  a reviewer-friendly ``reason`` string and the full numeric ``context``
  dict that produced it.

Why "candidate" and not "event": the cooldown layer (:mod:`.cooldown`)
gets a vote. A candidate becomes an :class:`watchagent.storage.Event`
only after the debouncer agrees this is the moment to fire. Keeping the
two layers separate is what lets the per-detector tests target the logic
("does the condition hold?") and the per-debouncer tests target the
suppression ("given it holds, do we emit?") independently.

Reason strings follow a consistent shape:

    {City} {observation} — {comparison to baseline / threshold}

so a reviewer reading ``events.reason`` sees the *number*, the
*comparison*, and the *yardstick* in one sentence. The same numbers live
in ``context`` for the API consumer.

The current six detectors:

1. :class:`TemperatureAnomalyDetector` — z-score vs trailing window, with
   an absolute warm-up fallback.
2. :class:`RapidTempChangeDetector`    — |ΔT|/h vs the prior reading.
3. :class:`WindDangerDetector`         — absolute wind speed threshold.
4. :class:`PrecipitationOnsetDetector` — 0 → >0 transition.
5. :class:`HeavyPrecipitationDetector` — sustained moderate/heavy bands.
6. :class:`WeatherCodeTransitionDetector` — escalation up the WMO tier order.
7. :class:`FeelsLikeDivergenceDetector` — apparent vs actual temperature.

(A cross-city outlier detector is intentionally deferred: it crosses
per-city state and has to align timestamps within a tolerance window,
which is a separate concern from these per-city ones. README documents
the design and lists it as future work.)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from watchagent.weather_codes import (
    description_of,
    is_more_severe,
    severity_of,
    tier_of,
)

if TYPE_CHECKING:
    from watchagent.detection.state import CityState
    from watchagent.storage import Reading


# Numeric epsilon for the std == 0 guard. Anything below this we treat as
# "the window is flat; the z-score is undefined". 1e-9 °C is well below
# Open-Meteo's reporting precision (0.1°C), so this only triggers on
# genuinely identical readings.
_STD_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    """A detector's verdict for a single reading.

    The engine wraps this in a real :class:`watchagent.storage.Event` (with
    city, reading_time, reading_time_utc, detected_at) only if the debouncer
    approves. ``severity`` is one of ``"low" | "medium" | "high"`` —
    matching :class:`watchagent.weather_codes.Severity` so the TEXT column
    values are consistent across detectors.
    """

    event_type: str
    severity: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    """Structural type for a detector.

    Anything with these two attributes is a detector. We use a Protocol
    rather than an ABC so the engine can accept any callable+attribute
    combination — including stubs in tests.
    """

    event_type: str

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        ...


# ---------------------------------------------------------------------------
# 1. Temperature anomaly (z-score with warm-up fallback)
# ---------------------------------------------------------------------------


class TemperatureAnomalyDetector:
    """Per-city temperature anomaly via rolling z-score.

    Design (the four points the reviewer flagged):

    * **Disjoint baseline.** We compute mean/std from
      ``state.temperatures()`` — the *prior* W readings — and test the
      current reading against that baseline. ``CityState`` guarantees the
      current reading is NOT in the window (the engine adds it after
      ``evaluate`` returns), so a genuine spike is not dampened by being
      folded into its own mean.
    * **Std-zero guard.** Overnight, a city can return several identical
      temperatures; ``pstdev`` then evaluates to 0. Dividing by it would
      produce inf/nan and either crash or fire a garbage event. When std
      is below :data:`_STD_EPSILON` we explicitly skip the z-score for
      this reading — the baseline is degenerate, not informative.
    * **Warm-up fallback.** Until the window has ``min_samples`` readings,
      the baseline isn't statistically meaningful. We fall back to absolute
      safety thresholds (configurable, default 35°C / −30°C) so a brand-new
      DB still detects extremes; the reason string says so explicitly.
    * **Reason carries the numbers.** ``"Ottawa 31.2°C is +3.1σ above its
      trailing-48 mean of 22.4°C (σ=2.8, threshold=±2.5σ)"`` — the same
      values populate ``context`` so the API consumer doesn't have to
      parse the prose.
    """

    event_type = "temperature_anomaly"

    def __init__(
        self,
        *,
        min_samples: int,
        z_thresh: float,
        warmup_high: float,
        warmup_low: float,
    ) -> None:
        if min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        if z_thresh <= 0:
            raise ValueError("z_thresh must be > 0")
        if warmup_low >= warmup_high:
            raise ValueError("warmup_low must be < warmup_high")
        self._min_samples = min_samples
        self._z_thresh = z_thresh
        self._warmup_high = warmup_high
        self._warmup_low = warmup_low

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        priors = state.temperatures()
        n = len(priors)

        if n < self._min_samples:
            return self._warmup_fallback(reading, n)

        mean = statistics.fmean(priors)
        # Sample std (Bessel's correction, ddof=1). The window is a *sample*
        # of the city's recent behavior, not its full population — the city
        # continues to exist outside the window. At W=48 the difference vs
        # population std is ~1%, but during the warm-up-to-full ramp
        # (n=6→48) sample std is meaningfully larger, which makes the
        # z-score slightly more conservative when the baseline is shortest.
        # statistics.stdev requires n ≥ 2, guaranteed by min_samples ≥ 2.
        std = statistics.stdev(priors)
        if std < _STD_EPSILON or not math.isfinite(std):
            # Flat / degenerate window — z-score undefined; skip.
            #
            # NOTE: this is a deliberate blind spot covered by the rate
            # detector. A flat baseline followed by a real jump produces
            # std=0 → no temperature_anomaly event, but rapid_temp_change
            # sees the same jump and fires on |ΔT|/h. The test
            # ``TestBackstopCoverage.test_flat_window_with_real_spike_is_caught_by_rate``
            # in test_detection_engine.py pins this property.
            return None

        z = (reading.temperature_2m - mean) / std
        if abs(z) < self._z_thresh:
            return None

        severity = self._severity_for_z(z)
        sign = "+" if z > 0 else ""
        direction = "above" if z > 0 else "below"
        reason = (
            f"{reading.city} {reading.temperature_2m:.1f}°C is {sign}{z:.1f}σ "
            f"{direction} its trailing-{n} mean of {mean:.1f}°C "
            f"(σ={std:.1f}, threshold=±{self._z_thresh:.1f}σ)"
        )
        return CandidateEvent(
            event_type=self.event_type,
            severity=severity,
            reason=reason,
            context={
                "method": "z_score",
                "value": reading.temperature_2m,
                "mean": mean,
                "std": std,
                "z": z,
                "z_threshold": self._z_thresh,
                "window_size": n,
            },
        )

    def _warmup_fallback(self, reading: Reading, n: int) -> CandidateEvent | None:
        t = reading.temperature_2m
        if t >= self._warmup_high:
            return CandidateEvent(
                event_type=self.event_type,
                severity="high",
                reason=(
                    f"{reading.city} {t:.1f}°C ≥ warm-up safety threshold "
                    f"{self._warmup_high:.1f}°C "
                    f"(only {n} prior readings, insufficient for z-score)"
                ),
                context={
                    "method": "warmup_absolute_high",
                    "value": t,
                    "threshold": self._warmup_high,
                    "window_size": n,
                    "min_samples": self._min_samples,
                },
            )
        if t <= self._warmup_low:
            return CandidateEvent(
                event_type=self.event_type,
                severity="high",
                reason=(
                    f"{reading.city} {t:.1f}°C ≤ warm-up safety threshold "
                    f"{self._warmup_low:.1f}°C "
                    f"(only {n} prior readings, insufficient for z-score)"
                ),
                context={
                    "method": "warmup_absolute_low",
                    "value": t,
                    "threshold": self._warmup_low,
                    "window_size": n,
                    "min_samples": self._min_samples,
                },
            )
        return None

    @staticmethod
    def _severity_for_z(z: float) -> str:
        a = abs(z)
        if a >= 4.0:
            return "high"
        if a >= 3.0:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# 2. Rapid temperature change
# ---------------------------------------------------------------------------


class RapidTempChangeDetector:
    """Fire when |ΔT| / hour vs the immediately prior reading exceeds a threshold.

    "Per hour" is computed from the actual UTC delta, not the cadence,
    so a missed cycle (e.g. one upstream failure) doesn't inflate the
    rate when readings span 2h. We use ``reading_time_utc`` (computed in
    M4 from ``current.time`` and ``utc_offset_seconds``) — local civil
    time would mis-compute deltas across DST.
    """

    event_type = "rapid_temp_change"

    def __init__(self, *, rate_thresh: float) -> None:
        if rate_thresh <= 0:
            raise ValueError("rate_thresh must be > 0")
        self._rate_thresh = rate_thresh

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        prev = state.last
        if prev is None:
            return None  # no baseline, can't compute a rate

        try:
            now_utc = datetime.fromisoformat(reading.reading_time_utc)
            prev_utc = datetime.fromisoformat(prev.reading_time_utc)
        except ValueError:
            # Defence in depth: if either timestamp is malformed, skip
            # rather than crash. The poller's defensive parse should
            # have rejected it already, but never trust upstream data.
            return None

        hours = (now_utc - prev_utc).total_seconds() / 3600.0
        if hours <= 0:
            return None  # non-monotonic timestamps; can't divide by a non-positive

        delta = reading.temperature_2m - prev.temperature_2m
        rate = delta / hours
        if abs(rate) < self._rate_thresh:
            return None

        severity = "high" if abs(rate) >= 2 * self._rate_thresh else "medium"
        sign = "+" if delta > 0 else ""
        reason = (
            f"{reading.city} temperature changed {sign}{delta:.1f}°C "
            f"over {hours:.1f}h ({sign}{rate:.1f}°C/h) — "
            f"exceeds rate threshold {self._rate_thresh:.1f}°C/h"
        )
        return CandidateEvent(
            event_type=self.event_type,
            severity=severity,
            reason=reason,
            context={
                "delta_celsius": delta,
                "elapsed_hours": hours,
                "rate_celsius_per_hour": rate,
                "threshold_celsius_per_hour": self._rate_thresh,
                "previous_temperature": prev.temperature_2m,
                "previous_reading_time_utc": prev.reading_time_utc,
            },
        )


# ---------------------------------------------------------------------------
# 3. Wind danger (absolute)
# ---------------------------------------------------------------------------


class WindDangerDetector:
    """Fire when ``wind_speed_10m`` (km/h) reaches a danger threshold.

    Severity bands roughly track Environment Canada's wind warnings:

    * ≥ threshold (default 40 km/h): low — "strong wind"
    * ≥ 60 km/h:                     medium — "advisory"
    * ≥ 80 km/h:                     high   — "warning"
    """

    event_type = "wind_danger"

    def __init__(self, *, threshold: float) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        self._threshold = threshold

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        wind = reading.wind_speed_10m
        if wind < self._threshold:
            return None

        if wind >= 80.0:
            severity = "high"
        elif wind >= 60.0:
            severity = "medium"
        else:
            severity = "low"

        reason = (
            f"{reading.city} wind {wind:.1f} km/h ≥ danger threshold "
            f"{self._threshold:.1f} km/h"
        )
        return CandidateEvent(
            event_type=self.event_type,
            severity=severity,
            reason=reason,
            context={
                "wind_kmh": wind,
                "threshold_kmh": self._threshold,
            },
        )


# ---------------------------------------------------------------------------
# 4. Precipitation onset (0 → >0)
# ---------------------------------------------------------------------------


class PrecipitationOnsetDetector:
    """Fire when precipitation transitions from exactly 0.0 mm/h to >0.

    The detector itself is edge-aware (it only returns a candidate at the
    moment of the transition), so the cooldown is mostly a redundancy
    against repeated transitions in a noisy sequence — but we keep the
    two layers separated for uniformity with the other detectors.
    """

    event_type = "precipitation_onset"

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        prev = state.last
        if prev is None:
            return None  # need a "before" to detect a transition
        if prev.precipitation > 0.0:
            return None  # was already wet
        if reading.precipitation <= 0.0:
            return None  # still dry

        reason = (
            f"{reading.city} precipitation began ({reading.precipitation:.2f} mm/h) — "
            f"first wet hour after a dry preceding hour "
            f"(prev {prev.reading_time_utc})"
        )
        return CandidateEvent(
            event_type=self.event_type,
            severity="low",
            reason=reason,
            context={
                "precipitation_mm_per_hour": reading.precipitation,
                "previous_precipitation_mm_per_hour": prev.precipitation,
                "previous_reading_time_utc": prev.reading_time_utc,
            },
        )


# ---------------------------------------------------------------------------
# 5. Heavy precipitation (intensity bands)
# ---------------------------------------------------------------------------


class HeavyPrecipitationDetector:
    """Fire on sustained heavy precipitation, banded by intensity.

    Independent of onset: a long, steadily heavy period can fire the
    intensity event without being on a 0→>0 boundary. Cooldown ensures
    a multi-hour storm gets one event per cooldown window, not one per
    reading.
    """

    event_type = "heavy_precipitation"

    def __init__(self, *, moderate_thresh: float, heavy_thresh: float) -> None:
        if moderate_thresh <= 0:
            raise ValueError("moderate_thresh must be > 0")
        if heavy_thresh < moderate_thresh:
            raise ValueError("heavy_thresh must be >= moderate_thresh")
        self._moderate = moderate_thresh
        self._heavy = heavy_thresh

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        precip = reading.precipitation
        if precip < self._moderate:
            return None

        if precip >= self._heavy:
            severity = "high"
            band = "heavy"
        else:
            severity = "medium"
            band = "moderate"

        reason = (
            f"{reading.city} {band} precipitation {precip:.2f} mm/h "
            f"≥ threshold {self._moderate:.1f} mm/h "
            f"(heavy band ≥ {self._heavy:.1f} mm/h)"
        )
        return CandidateEvent(
            event_type=self.event_type,
            severity=severity,
            reason=reason,
            context={
                "precipitation_mm_per_hour": precip,
                "moderate_threshold_mm_per_hour": self._moderate,
                "heavy_threshold_mm_per_hour": self._heavy,
                "band": band,
            },
        )


# ---------------------------------------------------------------------------
# 6. Weather code tier transition (escalation only)
# ---------------------------------------------------------------------------


class WeatherCodeTransitionDetector:
    """Fire when the WMO weather code escalates to a *more severe* tier.

    Within-tier intensity changes (e.g. "slight rain" → "heavy rain",
    both Tier RAIN) are detector #4/#5's job, not this one's. This
    detector explicitly routes through
    :func:`watchagent.weather_codes.is_more_severe`, which returns True
    only on strict tier upgrades. Downgrades and lateral moves don't
    fire — that's the "transition INTO an anomalous state" rule.
    """

    event_type = "weather_code_transition"

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        prev = state.last
        if prev is None:
            return None

        new_code = reading.weather_code
        old_code = prev.weather_code
        if not is_more_severe(new_code, old_code):
            return None

        new_tier = tier_of(new_code)
        old_tier = tier_of(old_code)
        new_label = description_of(new_code) or f"code {new_code}"
        old_label = description_of(old_code) or f"code {old_code}"
        # severity_of returns the WMO-tier severity (e.g. THUNDERSTORM = high)
        severity = str(severity_of(new_code) or "medium")

        reason = (
            f"{reading.city} weather escalated from "
            f"{old_label} ({old_tier.name if old_tier else 'UNKNOWN'}) to "
            f"{new_label} ({new_tier.name if new_tier else 'UNKNOWN'})"
        )
        return CandidateEvent(
            event_type=self.event_type,
            severity=severity,
            reason=reason,
            context={
                "from_code": old_code,
                "from_tier": old_tier.name if old_tier else None,
                "from_label": old_label,
                "to_code": new_code,
                "to_tier": new_tier.name if new_tier else None,
                "to_label": new_label,
            },
        )


# ---------------------------------------------------------------------------
# 7. Apparent-vs-actual divergence (wind chill / humidex)
# ---------------------------------------------------------------------------


class FeelsLikeDivergenceDetector:
    """Fire when ``|apparent_temperature − temperature_2m|`` exceeds a threshold.

    A 7°C gap between actual and apparent is the marker of dangerous
    wind chill (winter) or humidex (summer). Both directions matter, so
    we test the absolute difference; the sign goes into the reason
    string and the context dict.
    """

    event_type = "feels_like_divergence"

    def __init__(self, *, threshold: float) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        self._threshold = threshold

    def evaluate(self, reading: Reading, state: CityState) -> CandidateEvent | None:
        actual = reading.temperature_2m
        apparent = reading.apparent_temperature
        diff = apparent - actual
        abs_diff = abs(diff)
        if abs_diff < self._threshold:
            return None

        severity = "high" if abs_diff >= 10.0 else "medium"
        direction = "warmer" if diff > 0 else "cooler"
        reason = (
            f"{reading.city} feels {abs_diff:.1f}°C {direction} than the air "
            f"({apparent:.1f}°C apparent vs {actual:.1f}°C actual) — "
            f"threshold ±{self._threshold:.1f}°C"
        )
        return CandidateEvent(
            event_type=self.event_type,
            severity=severity,
            reason=reason,
            context={
                "apparent_celsius": apparent,
                "actual_celsius": actual,
                "diff_celsius": diff,
                "abs_diff_celsius": abs_diff,
                "threshold_celsius": self._threshold,
            },
        )
