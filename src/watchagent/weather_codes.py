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
  is safer than a false fire on data we don't recognise — but to keep the
  silence *observable*, the FIRST time an unknown code is seen we emit a
  one-shot INFO log (``weather_code.unknown``). Subsequent sightings of the
  same code are deduped via :data:`_seen_unknown_codes`.

Tier ordering rationale (CLEAR < … < THUNDERSTORM):
    CLEAR                   — clear sky
    CLOUDY                  — clouds, no precip
    FOG                     — visibility hazard but no precip
    DRIZZLE                 — light non-frozen precip
    RAIN                    — significant non-frozen precip; flooding risk
    SNOW                    — frozen precip + driving + cold-soak risk
    FREEZING_PRECIPITATION  — ice accretion: power-line and road glaze
    THUNDERSTORM            — precip + lightning + wind + hail risk

Why FREEZING_PRECIPITATION sits ABOVE SNOW for these three cities: in
Ottawa specifically, ice accretion (freezing rain / freezing drizzle) is
the most disruptive winter hazard category — the 1998 North American Ice
Storm is regional folklore, and even smaller events take down power lines
and shut transit faster than an equivalent volume of snow does. Toronto
sees less frequent but similar-magnitude events; Vancouver sees them
rarely but they are catastrophic when they happen because the city is
unprepared. Folding 56/57/66/67 into the plain RAIN tier (4) would
under-rate the worst case the system needs to surface promptly.

This ordering choice is the single place to revisit if domain assumptions
change (e.g. the system is later deployed to subtropical cities where
freezing precip is moot).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from watchagent.logging_setup import get_logger


class Tier(IntEnum):
    """Categorical severity tier with a strict ordering.

    Detector #5 fires when a reading transitions into a strictly higher tier
    (CLEAR → THUNDERSTORM, RAIN → FREEZING_PRECIPITATION, …). Comparing tiers
    via ``>`` answers exactly that question.
    """

    CLEAR = 0
    CLOUDY = 1
    FOG = 2
    DRIZZLE = 3
    RAIN = 4
    SNOW = 5
    FREEZING_PRECIPITATION = 6
    THUNDERSTORM = 7


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
# Completeness against the published Open-Meteo set is asserted by
# ``test_registry_covers_open_meteo_set`` so no code can be silently dropped
# by a refactor — silent gaps would become invisible dead spots in the
# detector-#5 transition logic.
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
    # Freezing precipitation: tier elevated above SNOW, severity HIGH across
    # all four codes — even "light" freezing drizzle can produce hazardous
    # ice accretion on roads and overhead lines. See module docstring.
    56: WeatherCode(56, "Light freezing drizzle",     Tier.FREEZING_PRECIPITATION, Severity.HIGH),
    57: WeatherCode(57, "Dense freezing drizzle",     Tier.FREEZING_PRECIPITATION, Severity.HIGH),
    61: WeatherCode(61, "Slight rain",                Tier.RAIN,         Severity.LOW),
    63: WeatherCode(63, "Moderate rain",              Tier.RAIN,         Severity.MEDIUM),
    65: WeatherCode(65, "Heavy rain",                 Tier.RAIN,         Severity.HIGH),
    66: WeatherCode(66, "Light freezing rain",        Tier.FREEZING_PRECIPITATION, Severity.HIGH),
    67: WeatherCode(67, "Heavy freezing rain",        Tier.FREEZING_PRECIPITATION, Severity.HIGH),
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


# The canonical Open-Meteo emit set — used by tests to detect drift if a code
# is added to (or dropped from) :data:`WMO_CODES` without updating the test.
OPEN_METEO_PUBLISHED_CODES: frozenset[int] = frozenset(
    {0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
     71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99}
)


# First-sighting cache for unknown codes. Module-level so it dedupes across
# every caller — first call to ``lookup(unknown)`` (or any helper that routes
# through it) emits one INFO line; subsequent calls are silent. Test code can
# call :func:`_clear_unknown_codes_cache` to reset between assertions.
_seen_unknown_codes: set[int] = set()


def _clear_unknown_codes_cache() -> None:
    """Reset the first-sighting log dedup cache. Test-only helper."""
    _seen_unknown_codes.clear()


def lookup(code: int) -> WeatherCode | None:
    """Return the :class:`WeatherCode` record for ``code`` or ``None`` if unknown.

    On the FIRST sighting of an unknown code, emits a one-shot INFO log so
    operators learn that Open-Meteo started emitting something we don't map.
    Subsequent sightings of the same code are silent.
    """
    rec = WMO_CODES.get(code)
    if rec is None and code not in _seen_unknown_codes:
        _seen_unknown_codes.add(code)
        get_logger(__name__).info(
            "weather_code.unknown",
            weather_code=code,
            note="not in WMO 4677 registry; tier-transition events suppressed",
        )
    return rec


def tier_of(code: int) -> Tier | None:
    rec = lookup(code)
    return rec.tier if rec is not None else None


def severity_of(code: int) -> Severity | None:
    rec = lookup(code)
    return rec.severity if rec is not None else None


def description_of(code: int) -> str:
    """Always return a human-readable label so log lines and event reasons stay readable.

    Falls back to ``"Unknown WMO code <n>"`` for codes Open-Meteo might add later.
    """
    rec = lookup(code)
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
