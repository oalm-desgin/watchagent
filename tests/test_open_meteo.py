"""Tests for the Open-Meteo HTTP client.

All HTTP traffic is mocked via ``respx`` — these tests must run with no
network access (the CI build job builds the image with no env vars and
no network, so any leaked real call would surface there).

The headline tests pin the M4 design decisions surfaced in review:
* ``test_fetch_current_4xx_does_not_retry`` — 4xx is permanent.
* ``test_fetch_current_5xx_retries_then_gives_up`` — 5xx is transient.
* ``test_fetch_current_timeout_retries_then_succeeds`` — timeouts retry.
* ``test_fetch_current_partial_payload_returns_none`` — null fields skip.
* ``test_fetch_current_uses_query_params_not_string_interpolation`` —
  proves the params dict round-trips through httpx URL-encoding cleanly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
import respx

from watchagent.cities import CITIES
from watchagent.open_meteo import OpenMeteoClient
from watchagent.storage import Reading

OTTAWA = CITIES[0]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Single AsyncClient per test — mirrors the lifespan-owned pattern."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
        yield c


@pytest.fixture
def fast_meteo(client: httpx.AsyncClient) -> OpenMeteoClient:
    """OpenMeteoClient with a tiny backoff base so tests don't sleep seconds."""
    return OpenMeteoClient(
        client=client,
        max_retries=3,
        backoff_base_seconds=0.001,
    )


def _good_payload(
    *,
    reading_time: str = "2026-05-28T13:00",
    utc_offset_seconds: int = -14400,
    temperature_2m: float = 21.5,
    apparent_temperature: float = 19.0,
    precipitation: float = 0.0,
    wind_speed_10m: float = 10.0,
    weather_code: int = 0,
) -> dict[str, object]:
    return {
        "latitude": 45.42,
        "longitude": -75.69,
        "utc_offset_seconds": utc_offset_seconds,
        "current": {
            "time": reading_time,
            "temperature_2m": temperature_2m,
            "apparent_temperature": apparent_temperature,
            "precipitation": precipitation,
            "wind_speed_10m": wind_speed_10m,
            "weather_code": weather_code,
        },
    }


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_happy_path(fast_meteo: OpenMeteoClient) -> None:
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=_good_payload())
    )
    reading = await fast_meteo.fetch_current(OTTAWA)

    assert reading is not None
    assert isinstance(reading, Reading)
    assert reading.city == "Ottawa"
    assert reading.reading_time == "2026-05-28T13:00"
    assert reading.reading_time_utc == "2026-05-28T17:00:00+00:00"
    assert reading.temperature_2m == 21.5
    assert reading.weather_code == 0
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_uses_query_params_not_string_interpolation(
    fast_meteo: OpenMeteoClient,
) -> None:
    """Build the URL via httpx params, never via f-string. Verifies the
    request landed with the five required `current` fields, the right
    coords, and `timezone=auto` — each as URL-encoded query params."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=_good_payload())
    )
    await fast_meteo.fetch_current(OTTAWA)

    req = route.calls.last.request
    qs = dict(req.url.params)
    assert qs["latitude"] == "45.42"
    assert qs["longitude"] == "-75.69"
    assert qs["wind_speed_unit"] == "kmh"
    assert qs["timezone"] == "auto"

    fields = qs["current"].split(",")
    assert set(fields) == {
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
    }
    # Defensive: no HTML entity bugs leaking into the wire URL.
    assert "&amp;" not in str(req.url)


# ---------------------------------------------------------------------------
# Retry policy: 4xx is PERMANENT, 5xx is TRANSIENT.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_4xx_does_not_retry(
    fast_meteo: OpenMeteoClient,
) -> None:
    """4xx is the request's fault — retrying just burns the cycle."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(400, json={"error": "bad request"})
    )
    result = await fast_meteo.fetch_current(OTTAWA)
    assert result is None
    assert route.call_count == 1, "4xx must be permanent — no retry"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_404_does_not_retry(
    fast_meteo: OpenMeteoClient,
) -> None:
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(404)
    )
    result = await fast_meteo.fetch_current(OTTAWA)
    assert result is None
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_5xx_retries_then_gives_up(
    fast_meteo: OpenMeteoClient,
) -> None:
    """503 every time → max_retries+1 attempts then None."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(503)
    )
    result = await fast_meteo.fetch_current(OTTAWA)
    assert result is None
    # max_retries=3 → 4 total attempts (initial + 3 retries).
    assert route.call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_5xx_then_200_succeeds(
    fast_meteo: OpenMeteoClient,
) -> None:
    """First attempt fails with 503; second returns 200 → reading parsed."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_good_payload()),
        ]
    )
    reading = await fast_meteo.fetch_current(OTTAWA)
    assert reading is not None
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_timeout_retries_then_succeeds(
    fast_meteo: OpenMeteoClient,
) -> None:
    """Timeout on attempt 1, success on attempt 2."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        side_effect=[
            httpx.TimeoutException("read timeout"),
            httpx.Response(200, json=_good_payload()),
        ]
    )
    reading = await fast_meteo.fetch_current(OTTAWA)
    assert reading is not None
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_network_error_retries_then_gives_up(
    fast_meteo: OpenMeteoClient,
) -> None:
    """ConnectError 4 times → None."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        side_effect=httpx.ConnectError("dns failure")
    )
    result = await fast_meteo.fetch_current(OTTAWA)
    assert result is None
    assert route.call_count == 4


# ---------------------------------------------------------------------------
# Partial / null payload handling.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_partial_payload_returns_none(
    fast_meteo: OpenMeteoClient,
) -> None:
    """If any of the 5 required fields is null, return None — no retry."""
    payload = _good_payload()
    payload["current"]["temperature_2m"] = None  # type: ignore[index]

    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    result = await fast_meteo.fetch_current(OTTAWA)
    assert result is None
    assert route.call_count == 1, (
        "Null fields are not retried — upstream is unlikely to fill the gap "
        "this cycle and we never store partial rows."
    )


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_zero_values_are_valid(
    fast_meteo: OpenMeteoClient,
) -> None:
    """precipitation=0.0 and weather_code=0 are CLEAR-sky, not null.
    Use `is None` checks, not falsy checks, when validating fields."""
    payload = _good_payload(precipitation=0.0, weather_code=0)
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    reading = await fast_meteo.fetch_current(OTTAWA)
    assert reading is not None
    assert reading.precipitation == 0.0
    assert reading.weather_code == 0


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_missing_current_object_returns_none(
    fast_meteo: OpenMeteoClient,
) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={"latitude": 45.42})
    )
    assert await fast_meteo.fetch_current(OTTAWA) is None


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_missing_utc_offset_returns_none(
    fast_meteo: OpenMeteoClient,
) -> None:
    payload = _good_payload()
    del payload["utc_offset_seconds"]
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    assert await fast_meteo.fetch_current(OTTAWA) is None


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_bad_json_returns_none(
    fast_meteo: OpenMeteoClient,
) -> None:
    """A 200 with non-JSON body is treated as permanent — no retry."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, text="<html>500 internal proxy error</html>")
    )
    result = await fast_meteo.fetch_current(OTTAWA)
    assert result is None
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_top_level_not_dict_returns_none(
    fast_meteo: OpenMeteoClient,
) -> None:
    """200 + valid JSON but the body is a string/list (not an object).
    Treated as malformed — permanent skip, no retry."""
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json="not an object")
    )
    assert await fast_meteo.fetch_current(OTTAWA) is None
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_missing_current_time_returns_none(
    fast_meteo: OpenMeteoClient,
) -> None:
    payload = _good_payload()
    del payload["current"]["time"]  # type: ignore[index]
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    assert await fast_meteo.fetch_current(OTTAWA) is None
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_non_numeric_temperature_returns_none(
    fast_meteo: OpenMeteoClient,
) -> None:
    """Defence in depth: even if Open-Meteo returns "NA" in a numeric slot,
    the parse must catch the conversion error and return None — never let
    a ValueError surface as poller.unhandled_exception."""
    payload = _good_payload()
    payload["current"]["temperature_2m"] = "NA"  # type: ignore[index]
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    assert await fast_meteo.fetch_current(OTTAWA) is None
    assert route.call_count == 1, "non-numeric is permanent — no retry"


# ---------------------------------------------------------------------------
# Cross-city UTC computation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_computes_utc_per_city_offset(
    fast_meteo: OpenMeteoClient,
) -> None:
    """13:00 local in Vancouver (offset -25200) -> 20:00 UTC."""
    payload = _good_payload(reading_time="2026-05-28T13:00", utc_offset_seconds=-25200)
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    vancouver = CITIES[2]
    reading = await fast_meteo.fetch_current(vancouver)
    assert reading is not None
    assert reading.reading_time_utc == "2026-05-28T20:00:00+00:00"
