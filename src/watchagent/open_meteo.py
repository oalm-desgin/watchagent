"""Open-Meteo client.

Wraps a long-lived ``httpx.AsyncClient`` (created and closed by the FastAPI
lifespan in M7) and exposes a single :meth:`fetch_current` that returns either
a parsed :class:`~watchagent.storage.Reading` or ``None`` on permanent failure.

Design rules the README will reference:

* **One client for the app lifetime.** The reviewer's steer: never construct
  ``httpx.AsyncClient`` per request — that throws away connection pooling and
  TLS reuse. The lifespan owns the client; this class just borrows it.

* **Retry only on TRANSIENT failures.**
    - Network errors (``httpx.ConnectError``, ``httpx.ReadError``) → retry.
    - Timeouts (``httpx.TimeoutException``) → retry.
    - 5xx responses → retry.
    - **4xx responses → permanent.** A 400 or 404 means the request is
      malformed or the resource is gone; retrying it three times just burns
      the cycle and the logs. Log once and skip.
    - JSON decode errors → permanent. The upstream is unlikely to start
      returning valid JSON within 4s.

  Backoff is exponential: ``backoff_base_seconds * 2**(attempt-1)``, so the
  default (1.0s base, 3 retries) gives the spec's 1s → 2s → 4s pattern.

* **Partial / null payload → permanent skip.** Open-Meteo occasionally
  returns ``null`` for an individual field. We never store partial rows
  (the DB column is ``NOT NULL`` regardless), and we never retry — the
  upstream is unlikely to fill the gap on the next attempt within the
  current cycle. Log at WARNING with the missing field names.

* **URL is built from a params dict, never string-interpolated.** ``httpx``
  does the URL-encoding so ``&`` / ``=`` / spaces never reach the wire as
  HTML entities or unencoded literals. (One of the §C C1 spec traps.)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx

from watchagent.cities import City
from watchagent.logging_setup import get_logger
from watchagent.storage import Reading, reading_time_to_utc, utc_now_iso

log = get_logger(__name__)


# Top-level Open-Meteo URL. Real-world hostname; respx mocks intercept this
# in tests so no network call is ever made during pytest.
DEFAULT_BASE_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoClient:
    """Async Open-Meteo wrapper. Receives a shared ``httpx.AsyncClient``.

    Parameters
    ----------
    client:
        Long-lived ``httpx.AsyncClient`` owned by the FastAPI lifespan. Its
        timeout configuration is the per-request timeout; we don't override
        it per-call.
    max_retries:
        Number of additional attempts after the initial request. Default 3
        gives 4 total attempts per fetch with backoff between each.
    backoff_base_seconds:
        Base for ``base * 2**(attempt-1)`` exponential backoff. Default 1.0
        produces the spec's 1s/2s/4s waits between attempts 1→2, 2→3, 3→4.
    sleep:
        Injectable async sleep so tests can run without real backoff delay.
    base_url:
        Override for tests / staging — not normally used.
    """

    REQUIRED_FIELDS: tuple[str, ...] = (
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
    )

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.client = client
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep
        self.base_url = base_url

    # ----- public ------------------------------------------------------

    async def fetch_current(self, city: City) -> Reading | None:
        """Fetch ``current`` conditions for ``city``.

        Returns a :class:`Reading` on success, ``None`` on permanent failure
        (4xx, exhausted retries, partial payload, JSON decode error). Never
        raises — the caller is :meth:`Poller._poll_city` which expects
        ``Reading | None`` and translates ``None`` into a status="error"
        outcome.
        """
        params = self._params_for(city)
        attempts = self.max_retries + 1
        last_failure: str = "unknown"

        for attempt in range(1, attempts + 1):
            try:
                response = await self.client.get(self.base_url, params=params)
            except httpx.TimeoutException as exc:
                last_failure = f"timeout: {type(exc).__name__}"
                log.warning(
                    "openmeteo.timeout",
                    city=city.name,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                if not await self._maybe_backoff(attempt, attempts, city):
                    log.warning(
                        "openmeteo.exhausted",
                        city=city.name,
                        attempts=attempt,
                        last_failure=last_failure,
                    )
                    return None
                continue
            except (httpx.ConnectError, httpx.ReadError, httpx.NetworkError) as exc:
                last_failure = f"network: {type(exc).__name__}"
                log.warning(
                    "openmeteo.network_error",
                    city=city.name,
                    attempt=attempt,
                    error=type(exc).__name__,
                    error_message=str(exc),
                )
                if not await self._maybe_backoff(attempt, attempts, city):
                    log.warning(
                        "openmeteo.exhausted",
                        city=city.name,
                        attempts=attempt,
                        last_failure=last_failure,
                    )
                    return None
                continue

            # 4xx — PERMANENT. Don't retry.
            if 400 <= response.status_code < 500:
                log.warning(
                    "openmeteo.client_error_permanent",
                    city=city.name,
                    http_status=response.status_code,
                    attempt=attempt,
                    note="4xx is not retried — request is malformed or resource gone",
                )
                return None

            # 5xx — TRANSIENT. Retry with backoff.
            if response.status_code >= 500:
                last_failure = f"http_{response.status_code}"
                log.warning(
                    "openmeteo.server_error",
                    city=city.name,
                    http_status=response.status_code,
                    attempt=attempt,
                )
                if not await self._maybe_backoff(attempt, attempts, city):
                    log.warning(
                        "openmeteo.exhausted",
                        city=city.name,
                        attempts=attempt,
                        last_failure=last_failure,
                    )
                    return None
                continue

            # 2xx — parse and return.
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                # Bad JSON is treated as permanent: retrying is unlikely to
                # help and we don't want to thrash on a misbehaving upstream.
                log.warning(
                    "openmeteo.bad_json",
                    city=city.name,
                    attempt=attempt,
                    error=str(exc),
                )
                return None

            return self._parse_payload(city, payload)

        # Defensive: every branch above either returns or `continue`s.
        return None

    # ----- internals ---------------------------------------------------

    def _params_for(self, city: City) -> dict[str, str | float]:
        """Build query params as a dict; httpx URL-encodes them safely.

        Never use ``f"...{x}..."`` here — string-interpolating params is the
        classic source of ``&amp;`` and unencoded-space bugs in spec §C1.
        """
        return {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "current": ",".join(self.REQUIRED_FIELDS),
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }

    async def _maybe_backoff(
        self, attempt: int, attempts: int, city: City
    ) -> bool:
        """Sleep for the next backoff window. Return ``True`` if the caller
        should retry, ``False`` if all attempts are exhausted."""
        if attempt >= attempts:
            return False
        wait = self.backoff_base_seconds * (2 ** (attempt - 1))
        log.warning(
            "openmeteo.retrying",
            city=city.name,
            wait_seconds=wait,
            next_attempt=attempt + 1,
            max_attempts=attempts,
        )
        await self._sleep(wait)
        return True

    def _parse_payload(self, city: City, payload: object) -> Reading | None:
        """Validate the response shape and produce a :class:`Reading`.

        Returns ``None`` (with a WARNING log) for any of:
            - missing/non-dict ``current``
            - missing/non-int ``utc_offset_seconds``
            - missing ``current.time``
            - any of the five required fields is ``None``

        ``precipitation = 0.0`` and ``weather_code = 0`` are valid; we use
        ``is None`` checks so falsy-but-meaningful values pass through.
        """
        if not isinstance(payload, dict):
            log.warning(
                "openmeteo.malformed_payload",
                city=city.name,
                reason="top_level_not_object",
            )
            return None

        current = payload.get("current")
        if not isinstance(current, dict):
            log.warning(
                "openmeteo.malformed_payload",
                city=city.name,
                reason="missing_current_object",
            )
            return None

        utc_offset = payload.get("utc_offset_seconds")
        if not isinstance(utc_offset, int):
            log.warning(
                "openmeteo.malformed_payload",
                city=city.name,
                reason="missing_utc_offset_seconds",
            )
            return None

        reading_time_local = current.get("time")
        if not isinstance(reading_time_local, str):
            log.warning(
                "openmeteo.malformed_payload",
                city=city.name,
                reason="missing_current_time",
            )
            return None

        # `is None` not `not …`: 0.0 precipitation and weather_code 0 are valid.
        missing_fields = [f for f in self.REQUIRED_FIELDS if current.get(f) is None]
        if missing_fields:
            log.warning(
                "openmeteo.partial_payload",
                city=city.name,
                missing_fields=missing_fields,
                reading_time=reading_time_local,
                note="not stored; upstream returned null for one or more fields",
            )
            return None

        # Final defence: if any numeric slot holds something we can't coerce
        # (e.g. "NA", a list, an object), treat the response as malformed —
        # permanent skip, not a retry. A misbehaving upstream is unlikely to
        # start returning numbers within the cycle, and a KeyError/ValueError
        # leaking out would surface as `poller.unhandled_exception`, which
        # wastes a slot in the gather safety net.
        try:
            return Reading(
                city=city.name,
                reading_time=reading_time_local,
                reading_time_utc=reading_time_to_utc(reading_time_local, utc_offset),
                fetched_at=utc_now_iso(),
                temperature_2m=float(current["temperature_2m"]),
                apparent_temperature=float(current["apparent_temperature"]),
                precipitation=float(current["precipitation"]),
                wind_speed_10m=float(current["wind_speed_10m"]),
                weather_code=int(current["weather_code"]),
            )
        except (TypeError, ValueError) as exc:
            log.warning(
                "openmeteo.malformed_payload",
                city=city.name,
                reason="non_numeric_field_value",
                error=str(exc),
                reading_time=reading_time_local,
            )
            return None
