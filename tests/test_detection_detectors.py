"""Per-detector unit tests.

Each detector is exercised in isolation, against a hand-built CityState.
No DB, no clock, no debouncer — just "given these priors and this
reading, did the condition hold?". The four design imperatives are pinned
explicitly:

* Disjoint baseline (the current reading is NOT in the window when
  ``evaluate`` is called).
* Std-zero guard (a flat window does NOT divide by zero).
* Warm-up fallback (small window → absolute thresholds, with the reason
  string explicitly noting "insufficient for z-score").
* Reason strings carry the numbers (and the same numbers populate
  ``context``).
"""

from __future__ import annotations

import pytest

from watchagent.detection.detectors import (
    FeelsLikeDivergenceDetector,
    HeavyPrecipitationDetector,
    PrecipitationOnsetDetector,
    RapidTempChangeDetector,
    TemperatureAnomalyDetector,
    WeatherCodeTransitionDetector,
    WindDangerDetector,
)
from watchagent.detection.state import CityState
from watchagent.storage import Reading


def make_reading(
    *,
    city: str = "Ottawa",
    reading_time_utc: str = "2026-05-28T12:00:00+00:00",
    temperature: float = 20.0,
    apparent: float | None = None,
    precipitation: float = 0.0,
    wind: float = 10.0,
    weather_code: int = 0,
) -> Reading:
    """Test factory. Defaults are deliberately benign: nothing fires
    unless the test changes the relevant field."""
    return Reading(
        city=city,
        reading_time=reading_time_utc.split("+", 1)[0],
        reading_time_utc=reading_time_utc,
        fetched_at="2026-05-28T12:00:30+00:00",
        temperature_2m=temperature,
        apparent_temperature=apparent if apparent is not None else temperature,
        precipitation=precipitation,
        wind_speed_10m=wind,
        weather_code=weather_code,
    )


def state_with(*priors: Reading, capacity: int = 48) -> CityState:
    s = CityState(city=priors[0].city if priors else "Ottawa", capacity=capacity)
    s.hydrate(priors)
    return s


# ---------------------------------------------------------------------------
# Temperature anomaly (z-score with warm-up fallback)
# ---------------------------------------------------------------------------


@pytest.fixture
def z_detector() -> TemperatureAnomalyDetector:
    return TemperatureAnomalyDetector(
        min_samples=6,
        z_thresh=2.5,
        warmup_high=35.0,
        warmup_low=-30.0,
    )


class TestTemperatureAnomalyZScore:
    def test_validators_reject_bad_args(self) -> None:
        with pytest.raises(ValueError):
            TemperatureAnomalyDetector(
                min_samples=1, z_thresh=2.5, warmup_high=35.0, warmup_low=-30.0
            )
        with pytest.raises(ValueError):
            TemperatureAnomalyDetector(
                min_samples=6, z_thresh=0, warmup_high=35.0, warmup_low=-30.0
            )
        with pytest.raises(ValueError):
            # warmup_low must be < warmup_high
            TemperatureAnomalyDetector(
                min_samples=6, z_thresh=2.5, warmup_high=10.0, warmup_low=10.0
            )

    def test_normal_reading_does_not_fire(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        priors = [make_reading(temperature=20.0 + i * 0.1) for i in range(10)]
        state = state_with(*priors)
        result = z_detector.evaluate(make_reading(temperature=20.5), state)
        assert result is None

    def test_fires_on_high_z_above(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        """6 priors, all 20°C ± small noise. Reading 30°C has |z| huge."""
        priors = [make_reading(temperature=20.0 + i * 0.1) for i in range(6)]
        state = state_with(*priors)
        result = z_detector.evaluate(make_reading(temperature=30.0), state)
        assert result is not None
        assert result.event_type == "temperature_anomaly"
        assert result.context["method"] == "z_score"
        assert result.context["z"] > 0
        assert result.context["window_size"] == 6
        assert "above" in result.reason
        # Reason carries the exact numbers from context — the standout sentence.
        assert f"{result.context['mean']:.1f}" in result.reason
        assert f"{result.context['std']:.1f}" in result.reason

    def test_fires_on_low_z_below(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        priors = [make_reading(temperature=20.0 + i * 0.1) for i in range(6)]
        state = state_with(*priors)
        result = z_detector.evaluate(make_reading(temperature=10.0), state)
        assert result is not None
        assert result.context["z"] < 0
        assert "below" in result.reason

    def test_severity_bands(self, z_detector: TemperatureAnomalyDetector) -> None:
        """Pin the |z| → severity mapping."""
        priors = [make_reading(temperature=20.0 + i * 0.1) for i in range(20)]
        state = state_with(*priors)

        # Massive z → high
        out_high = z_detector.evaluate(make_reading(temperature=50.0), state)
        assert out_high is not None and out_high.severity == "high"

    def test_disjoint_baseline_invariant(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        """The current reading must NOT be in the mean/std baseline.

        20 priors at exactly 20.0; current reading is 30.0. If the current
        were folded into the baseline, the mean would shift by 0.5 and std
        would be non-zero — z would be much smaller than reality. With the
        disjoint baseline, std==0 from the priors → we hit the std-zero
        guard and skip cleanly. Either way the implementation must NOT
        return a "z = ~9 above mean of 20.5" event with the current point
        contaminating the baseline.
        """
        priors = [make_reading(temperature=20.0) for _ in range(20)]
        state = state_with(*priors)
        result = z_detector.evaluate(make_reading(temperature=30.0), state)
        # priors are perfectly flat ⇒ std = 0 ⇒ z undefined ⇒ skip.
        # The ALTERNATIVE — "include current in baseline so std becomes
        # non-zero" — is exactly the bug the brief warned against, and
        # pinning that this returns None is how we keep the disjoint
        # invariant honest.
        assert result is None


class TestStdZeroGuard:
    def test_flat_window_does_not_divide_by_zero(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        """A flat window (overnight, 6 identical hourly readings) makes
        std==0. We must skip this reading, not crash and not fire a
        garbage event."""
        priors = [make_reading(temperature=22.0) for _ in range(6)]
        state = state_with(*priors)
        result = z_detector.evaluate(make_reading(temperature=22.0), state)
        assert result is None

    def test_flat_window_with_outlier_current_still_skips(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        """Even with a clearly-anomalous current reading, std==0 means the
        baseline is degenerate; we cannot answer "how anomalous?" honestly.
        Skip is the safe call."""
        priors = [make_reading(temperature=22.0) for _ in range(8)]
        state = state_with(*priors)
        result = z_detector.evaluate(make_reading(temperature=40.0), state)
        assert result is None


class TestWarmupFallback:
    def test_below_min_samples_normal_temp_does_not_fire(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        priors = [make_reading(temperature=20.0)]
        state = state_with(*priors)
        result = z_detector.evaluate(make_reading(temperature=22.0), state)
        assert result is None

    def test_below_min_samples_extreme_high_fires_on_absolute(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        state = CityState(city="Ottawa", capacity=48)  # zero priors
        result = z_detector.evaluate(make_reading(temperature=35.5), state)
        assert result is not None
        assert result.severity == "high"
        assert result.context["method"] == "warmup_absolute_high"
        assert "warm-up safety threshold" in result.reason
        assert "insufficient for z-score" in result.reason

    def test_below_min_samples_extreme_low_fires_on_absolute(
        self, z_detector: TemperatureAnomalyDetector
    ) -> None:
        state = CityState(city="Ottawa", capacity=48)
        result = z_detector.evaluate(make_reading(temperature=-32.0), state)
        assert result is not None
        assert result.context["method"] == "warmup_absolute_low"


# ---------------------------------------------------------------------------
# Rapid temperature change
# ---------------------------------------------------------------------------


class TestRapidTempChange:
    def test_no_priors_does_not_fire(self) -> None:
        d = RapidTempChangeDetector(rate_thresh=4.0)
        state = CityState(city="Ottawa", capacity=4)
        assert d.evaluate(make_reading(temperature=20.0), state) is None

    def test_below_rate_does_not_fire(self) -> None:
        d = RapidTempChangeDetector(rate_thresh=4.0)
        prev = make_reading(reading_time_utc="2026-05-28T11:00:00+00:00", temperature=20.0)
        state = state_with(prev)
        curr = make_reading(reading_time_utc="2026-05-28T12:00:00+00:00", temperature=22.0)
        assert d.evaluate(curr, state) is None

    def test_above_rate_fires(self) -> None:
        d = RapidTempChangeDetector(rate_thresh=4.0)
        prev = make_reading(reading_time_utc="2026-05-28T11:00:00+00:00", temperature=20.0)
        state = state_with(prev)
        curr = make_reading(reading_time_utc="2026-05-28T12:00:00+00:00", temperature=25.0)
        result = d.evaluate(curr, state)
        assert result is not None
        assert result.event_type == "rapid_temp_change"
        assert result.context["rate_celsius_per_hour"] == pytest.approx(5.0)
        assert result.severity == "medium"

    def test_double_rate_is_high_severity(self) -> None:
        d = RapidTempChangeDetector(rate_thresh=4.0)
        prev = make_reading(reading_time_utc="2026-05-28T11:00:00+00:00", temperature=20.0)
        state = state_with(prev)
        curr = make_reading(reading_time_utc="2026-05-28T12:00:00+00:00", temperature=29.0)
        result = d.evaluate(curr, state)
        assert result is not None
        assert result.severity == "high"

    def test_rate_uses_actual_elapsed_time_not_cadence(self) -> None:
        """If we missed a cycle, the elapsed gap is 2h — the same delta
        spread over 2h is half the rate. Must not over-fire from the
        missed cycle alone."""
        d = RapidTempChangeDetector(rate_thresh=4.0)
        prev = make_reading(reading_time_utc="2026-05-28T10:00:00+00:00", temperature=20.0)
        state = state_with(prev)
        # Same +5°C delta, but over 2 hours → 2.5°C/h, below threshold.
        curr = make_reading(reading_time_utc="2026-05-28T12:00:00+00:00", temperature=25.0)
        assert d.evaluate(curr, state) is None

    def test_non_monotonic_timestamps_do_not_crash(self) -> None:
        """Defence: if (somehow) the new reading is older than the last
        one in the window, hours <= 0 and we skip rather than divide
        by a non-positive."""
        d = RapidTempChangeDetector(rate_thresh=4.0)
        prev = make_reading(reading_time_utc="2026-05-28T13:00:00+00:00", temperature=20.0)
        state = state_with(prev)
        curr = make_reading(reading_time_utc="2026-05-28T12:00:00+00:00", temperature=30.0)
        assert d.evaluate(curr, state) is None


# ---------------------------------------------------------------------------
# Wind danger
# ---------------------------------------------------------------------------


class TestWindDanger:
    @pytest.fixture
    def detector(self) -> WindDangerDetector:
        return WindDangerDetector(threshold=40.0)

    def test_below_threshold_does_not_fire(self, detector: WindDangerDetector) -> None:
        state = CityState(city="Ottawa", capacity=4)
        assert detector.evaluate(make_reading(wind=39.9), state) is None

    @pytest.mark.parametrize(
        ("wind", "expected_severity"),
        [
            (40.0, "low"),
            (59.9, "low"),
            (60.0, "medium"),
            (79.9, "medium"),
            (80.0, "high"),
            (120.0, "high"),
        ],
    )
    def test_severity_bands(
        self,
        detector: WindDangerDetector,
        wind: float,
        expected_severity: str,
    ) -> None:
        state = CityState(city="Ottawa", capacity=4)
        result = detector.evaluate(make_reading(wind=wind), state)
        assert result is not None
        assert result.severity == expected_severity
        assert result.context["wind_kmh"] == wind


# ---------------------------------------------------------------------------
# Precipitation onset
# ---------------------------------------------------------------------------


class TestPrecipitationOnset:
    def test_no_priors_does_not_fire(self) -> None:
        d = PrecipitationOnsetDetector()
        state = CityState(city="Ottawa", capacity=4)
        assert d.evaluate(make_reading(precipitation=0.5), state) is None

    def test_dry_to_wet_fires(self) -> None:
        d = PrecipitationOnsetDetector()
        prev = make_reading(precipitation=0.0)
        state = state_with(prev)
        curr = make_reading(precipitation=0.4)
        result = d.evaluate(curr, state)
        assert result is not None
        assert result.event_type == "precipitation_onset"
        assert "began" in result.reason

    def test_wet_to_wet_does_not_fire(self) -> None:
        d = PrecipitationOnsetDetector()
        prev = make_reading(precipitation=0.5)
        state = state_with(prev)
        curr = make_reading(precipitation=0.7)
        assert d.evaluate(curr, state) is None

    def test_dry_to_dry_does_not_fire(self) -> None:
        d = PrecipitationOnsetDetector()
        prev = make_reading(precipitation=0.0)
        state = state_with(prev)
        curr = make_reading(precipitation=0.0)
        assert d.evaluate(curr, state) is None


# ---------------------------------------------------------------------------
# Heavy precipitation
# ---------------------------------------------------------------------------


class TestHeavyPrecipitation:
    @pytest.fixture
    def detector(self) -> HeavyPrecipitationDetector:
        return HeavyPrecipitationDetector(moderate_thresh=4.0, heavy_thresh=10.0)

    def test_validators(self) -> None:
        with pytest.raises(ValueError):
            HeavyPrecipitationDetector(moderate_thresh=0, heavy_thresh=10.0)
        with pytest.raises(ValueError):
            HeavyPrecipitationDetector(moderate_thresh=10.0, heavy_thresh=4.0)

    @pytest.mark.parametrize(
        ("precip", "should_fire", "expected_severity"),
        [
            (0.0, False, None),
            (3.9, False, None),
            (4.0, True, "medium"),
            (9.9, True, "medium"),
            (10.0, True, "high"),
            (50.0, True, "high"),
        ],
    )
    def test_intensity_bands(
        self,
        detector: HeavyPrecipitationDetector,
        precip: float,
        should_fire: bool,
        expected_severity: str | None,
    ) -> None:
        state = CityState(city="Ottawa", capacity=4)
        result = detector.evaluate(make_reading(precipitation=precip), state)
        if should_fire:
            assert result is not None
            assert result.severity == expected_severity
        else:
            assert result is None


# ---------------------------------------------------------------------------
# Weather code transition
# ---------------------------------------------------------------------------


class TestWeatherCodeTransition:
    def test_no_priors_does_not_fire(self) -> None:
        d = WeatherCodeTransitionDetector()
        state = CityState(city="Ottawa", capacity=4)
        assert d.evaluate(make_reading(weather_code=63), state) is None

    def test_clear_to_thunderstorm_fires(self) -> None:
        """0 (CLEAR) → 95 (THUNDERSTORM) — strict tier upgrade."""
        d = WeatherCodeTransitionDetector()
        prev = make_reading(weather_code=0)
        state = state_with(prev)
        curr = make_reading(weather_code=95)
        result = d.evaluate(curr, state)
        assert result is not None
        assert result.event_type == "weather_code_transition"
        assert result.context["from_code"] == 0
        assert result.context["to_code"] == 95
        assert result.severity == "high"

    def test_within_tier_does_not_fire(self) -> None:
        """61 (slight rain) → 65 (heavy rain) — both Tier RAIN, intensity
        change is detector #4/#5's job, not this one's."""
        d = WeatherCodeTransitionDetector()
        prev = make_reading(weather_code=61)
        state = state_with(prev)
        curr = make_reading(weather_code=65)
        assert d.evaluate(curr, state) is None

    def test_downgrade_does_not_fire(self) -> None:
        """Spec says 'transition INTO an anomalous state'. A downgrade is
        the opposite — never fire."""
        d = WeatherCodeTransitionDetector()
        prev = make_reading(weather_code=95)  # thunderstorm
        state = state_with(prev)
        curr = make_reading(weather_code=0)  # clear
        assert d.evaluate(curr, state) is None

    def test_freezing_precip_is_severe_upgrade(self) -> None:
        """61 (RAIN) → 67 (FREEZING_PRECIPITATION). Justified in M2 as
        the dominant Canadian-winter hazard."""
        d = WeatherCodeTransitionDetector()
        prev = make_reading(weather_code=61)
        state = state_with(prev)
        curr = make_reading(weather_code=67)
        result = d.evaluate(curr, state)
        assert result is not None
        assert result.severity == "high"


# ---------------------------------------------------------------------------
# Feels-like divergence
# ---------------------------------------------------------------------------


class TestFeelsLikeDivergence:
    @pytest.fixture
    def detector(self) -> FeelsLikeDivergenceDetector:
        return FeelsLikeDivergenceDetector(threshold=5.0)

    def test_below_threshold_does_not_fire(
        self, detector: FeelsLikeDivergenceDetector
    ) -> None:
        state = CityState(city="Ottawa", capacity=4)
        result = detector.evaluate(
            make_reading(temperature=10.0, apparent=14.9), state
        )
        assert result is None

    def test_negative_diff_fires_with_cooler(
        self, detector: FeelsLikeDivergenceDetector
    ) -> None:
        """Wind chill direction (apparent < actual) — the classic Canadian
        winter case the threshold is calibrated for."""
        state = CityState(city="Ottawa", capacity=4)
        result = detector.evaluate(
            make_reading(temperature=4.2, apparent=-3.0), state
        )
        assert result is not None
        assert "cooler" in result.reason
        assert result.context["diff_celsius"] < 0
        assert result.severity == "medium"

    def test_large_divergence_is_high_severity(
        self, detector: FeelsLikeDivergenceDetector
    ) -> None:
        state = CityState(city="Ottawa", capacity=4)
        result = detector.evaluate(
            make_reading(temperature=0.0, apparent=-12.0), state
        )
        assert result is not None
        assert result.severity == "high"

    def test_validators(self) -> None:
        with pytest.raises(ValueError):
            FeelsLikeDivergenceDetector(threshold=0)
