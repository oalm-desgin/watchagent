"""WMO weather code registry — single source of truth.

Open-Meteo returns an integer ``weather_code`` per reading per the WMO 4677
table (see https://open-meteo.com/en/docs). Centralising the mapping in one
module means:

* Detector #5 (weather-code transition) compares categorical tiers via a
  STRICT ORDERING: it fires only when ``new_tier > old_tier``. ``Tier`` is an
  ``IntEnum`` exactly so this comparison is well-defined — strings or ad-hoc
  category names would not be.
* Within-tier intensity changes (e.g. slight rain → heavy rain) are NOT a
  detector-#5 concern; precipitation/wind absolute detectors handle that.
* Unknown codes (Open-Meteo could ship a new code one day) return ``None``
  from the lookup helpers and ``False`` from :func:`is_more_severe`. Silence
  is safer than a false fire on data we don't recognise.

Severity ordering rationale (CLEAR < … < THUNDERSTORM):
    CLEAR        — clear sky
    CLOUDY       — clouds, no precip
    FOG          — visibility hazard but no precip
    DRIZZLE      — light precip
    RAIN         — significant precip; flooding/road-water risk
    SNOW         — precip + driving + freeze risk (Canadian context)
    THUNDERSTORM — precip + lightning + wind + hail risk

Reasonable people could swap RAIN/SNOW; we put SNOW above RAIN because in the
three monitored cities (all northern Canadian) snow is the more disruptive
hazard category. The IntEnum is the single place to revisit if that call
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Tier(IntEnum):
    """Categorical severity tier with a strict ordering.

    Detector #5 fires when a reading transitions into a strictly higher tier
    (CLEAR → THUNDERSTORM, CLOUDY → SNOW, etc.). Comparing tiers via ``>``
    answers exactly that question.
    """

    CLEAR = 0
    CLOUDY = 1
    FOG = 2
    DRIZZLE = 3
    RAIN = 4
    SNOW = 5
    THUNDERSTORM = 6


class Severity(StrEnum):
    """Human-graded severity that lands on the event row."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class WeatherCode:
    code: int
    description: str
    tier: Tier
    severity: Severity


# WMO 4677 codes Open-Meteo publishes. Source: https://open-meteo.com/en/docs.
WMO_CODES: dict[int, WeatherCode] = {
    0:  WeatherCode(0,  "Clear sky",                  Tier.CLEAR,        Severity.LOW),
    1:  WeatherCode(1,  "Mainly clear",               Tier.CLOUDY,       Severity.LOW),
    2:  WeatherCode(2,  "Partly cloudy",              Tier.CLOUDY,       Severity.LOW),
    3:  WeatherCode(3,  "Overcast",                   Tier.CLOUDY,       Severity.LOW),
    45: WeatherCode(45, "Fog",                        Tier.FOG,          Severity.MEDIUM),
    48: WeatherCode(48, "Depositing rime fog",        Tier.FOG,          Severity.MEDIUM),
    51: WeatherCode(51, "Light drizzle",              Tier.DRIZZLE,      Severity.LOW),
    53: WeatherCode(53, "Moderate drizzle",           Tier.DRIZZLE,      Severity.LOW),
    55: WeatherCode(55, "Dense drizzle",              Tier.DRIZZLE,      Severity.MEDIUM),
    56: WeatherCode(56, "Light freezing drizzle",     Tier.DRIZZLE,      Severity.MEDIUM),
    57: WeatherCode(57, "Dense freezing drizzle",     Tier.DRIZZLE,      Severity.HIGH),
    61: WeatherCode(61, "Slight rain",                Tier.RAIN,         Severity.LOW),
    63: WeatherCode(63, "Moderate rain",              Tier.RAIN,         Severity.MEDIUM),
    65: WeatherCode(65, "Heavy rain",                 Tier.RAIN,         Severity.HIGH),
    66: WeatherCode(66, "Light freezing rain",        Tier.RAIN,         Severity.HIGH),
    67: WeatherCode(67, "Heavy freezing rain",        Tier.RAIN,         Severity.HIGH),
    71: WeatherCode(71, "Slight snow fall",           Tier.SNOW,         Severity.LOW),
    73: WeatherCode(73, "Moderate snow fall",         Tier.SNOW,         Severity.MEDIUM),
    75: WeatherCode(75, "Heavy snow fall",            Tier.SNOW,         Severity.HIGH),
    77: WeatherCode(77, "Snow grains",                Tier.SNOW,         Severity.LOW),
    80: WeatherCode(80, "Slight rain showers",        Tier.RAIN,         Severity.LOW),
    81: WeatherCode(81, "Moderate rain showers",      Tier.RAIN,         Severity.MEDIUM),
    82: WeatherCode(82, "Violent rain showers",       Tier.RAIN,         Severity.HIGH),
    85: WeatherCode(85, "Slight snow showers",        Tier.SNOW,         Severity.LOW),
    86: WeatherCode(86, "Heavy snow showers",         Tier.SNOW,         Severity.HIGH),
    95: WeatherCode(95, "Thunderstorm",               Tier.THUNDERSTORM, Severity.HIGH),
    96: WeatherCode(96, "Thunderstorm with slight hail", Tier.THUNDERSTORM, Severity.HIGH),
    99: WeatherCode(99, "Thunderstorm with heavy hail",  Tier.THUNDERSTORM, Severity.HIGH),
}


def lookup(code: int) -> WeatherCode | None:
    """Return the :class:`WeatherCode` record for ``code`` or ``None`` if unknown."""
    return WMO_CODES.get(code)


def tier_of(code: int) -> Tier | None:
    rec = WMO_CODES.get(code)
    return rec.tier if rec is not None else None


def severity_of(code: int) -> Severity | None:
    rec = WMO_CODES.get(code)
    return rec.severity if rec is not None else None


def description_of(code: int) -> str:
    """Always return a human-readable label so log lines and event reasons stay readable.

    Falls back to ``"Unknown WMO code <n>"`` for codes Open-Meteo might add later.
    """
    rec = WMO_CODES.get(code)
    return rec.description if rec is not None else f"Unknown WMO code {code}"


def is_more_severe(new_code: int, old_code: int) -> bool:
    """Return True iff ``new_code`` belongs to a strictly higher tier than ``old_code``.

    Detector #5's firing condition. Returns False if either code is unknown —
    we'd rather miss an event than fire one on data we can't classify.
    """
    new_tier = tier_of(new_code)
    old_tier = tier_of(old_code)
    if new_tier is None or old_tier is None:
        return False
    return new_tier > old_tier
