"""End-to-end FIRE-path integration tests for M9.

The M7 dress rehearsal proved the no-fire / dedup path end-to-end and
visibly: ``new=3 -> duplicate=3`` across consecutive cycles is the
entire dedup chain printed in one log. This file does the symmetric
thing for the FIRE path:

* Mock Open-Meteo via ``respx`` so no network call is made.
* Boot the full FastAPI app via its lifespan, with the poller enabled.
* (Optionally) pre-seed the DB with warm-up readings so a single real
  poll cycle clears the z-score warm-up gate.
* Let the lifespan run real cycles. Wait until the expected number of
  rows have been inserted.
* GET ``/events`` through an in-process ASGI client and assert the
  events surface with the correct ``event_type``, ``severity``, AND a
  ``reason`` string carrying the numeric context that produced them.

Tests in this file:

1. ``TestEndToEndFirePath``
   The centerpiece. Fires a ``temperature_anomaly`` event via a real
   poller cycle and asserts the full record at ``/events``.

2. ``TestMultiDetectorOnOneReading``
   Integration variant of the M5 unit-test guarantee that a single
   reading can fire multiple detectors. Engineers a payload that trips
   ``temperature_anomaly`` + ``wind_danger`` +
   ``weather_code_transition`` simultaneously and asserts all three
   appear in ``/events`` for the same ``reading_time_utc``.

3. ``TestUrlQueryParamsAtIntegration``
   The §C C1 trap (``&amp;`` HTML-entity bug) caught at the wired
   level. Inspects the request that respx received during a real
   lifespan-driven cycle and asserts the query string round-tripped as
   decoded params. Locks down "the request that actually went on the
   wire" rather than just "the params dict".

4. ``TestCooldownAcrossRealRestart``
   Boots a real lifespan, fires an event, shuts down cleanly. Boots a
   SECOND real lifespan against the same DB file. Re-feeds the same
   condition. Asserts ``events_stored == 1`` -- the cooldown survived
   the restart through hydration, not through an in-memory artifact.
   The "events that never stop firing across restarts" failure mode
   the spec calls out, proven against the actual lifecycle.

Per-test isolation
==================

Every test takes ``tmp_path`` (pytest builtin, per-test directory).
``DB_PATH`` is constructed under ``tmp_path``, so tests that seed
events / readings cannot bleed into a later test's ``/health`` count
or ``/events`` listing. The cross-restart test creates its second
``Settings`` with the SAME ``db_path`` deliberately - that's the
point - but still inside its own ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
from httpx import ASGITransport, AsyncClient

from watchagent.api.app import create_app
from watchagent.cities import CITIES
from watchagent.config import Settings
from watchagent.storage import Database, Reading, reading_time_to_utc, utc_now_iso

OTTAWA, TORONTO, VANCOUVER = CITIES

# Open-Meteo URL the OpenMeteoClient hits. Mocking exactly this URL
# ensures we never make a real network call during pytest.
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# Helpers (pure functions / coroutines, NOT fixtures)
# ---------------------------------------------------------------------------


def _payload(
    *,
    city: tuple[float, float],
    reading_time: str,
    utc_offset_seconds: int = -14400,
    temperature_2m: float = 20.0,
    apparent_temperature: float | None = None,
    precipitation: float = 0.0,
    wind_speed_10m: float = 10.0,
    weather_code: int = 0,
) -> dict[str, Any]:
    """Build a realistic Open-Meteo response body.

    ``city`` is ``(latitude, longitude)`` so we can pin per-city payloads
    by the same coords the OpenMeteoClient sends in the query string.
    ``apparent_temperature`` defaults to ``temperature_2m`` so the
    feels-like-divergence detector stays quiet unless we explicitly
    diverge it.
    """
    lat, lon = city
    return {
        "latitude": lat,
        "longitude": lon,
        "utc_offset_seconds": utc_offset_seconds,
        "current": {
            "time": reading_time,
            "temperature_2m": temperature_2m,
            "apparent_temperature": (
                temperature_2m if apparent_temperature is None
                else apparent_temperature
            ),
            "precipitation": precipitation,
            "wind_speed_10m": wind_speed_10m,
            "weather_code": weather_code,
        },
    }


def _make_per_city_responder(
    payloads_by_lat: dict[float, dict[str, Any]],
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a respx ``side_effect`` that picks payloads by city latitude.

    The OpenMeteoClient sends ``latitude`` as the first query param. We
    parse it back to a float and look up the right payload. Any city we
    didn't script returns a benign zero-everything payload so the poller
    doesn't break on Toronto/Vancouver while we're targeting Ottawa.
    """
    keys = {round(k, 4): v for k, v in payloads_by_lat.items()}

    def respond(request: httpx.Request) -> httpx.Response:
        lat = round(float(request.url.params["latitude"]), 4)
        payload = keys.get(lat)
        if payload is None:
            # Fallback: a totally benign reading at the requested coords,
            # with a never-overlapping time so dedup-or-no doesn't matter.
            payload = _payload(
                city=(lat, float(request.url.params["longitude"])),
                reading_time="2000-01-01T00:00",
                temperature_2m=10.0,
            )
        return httpx.Response(200, json=payload)

    return respond


async def _seed_warmup_readings(
    db_path: str,
    *,
    city: str,
    count: int,
    base_time: datetime,
    base_temp: float = 20.0,
    interval_hours: int = 1,
    weather_code: int = 0,
    wind_speed: float = 10.0,
) -> list[Reading]:
    """Seed ``count`` warm-up readings into the DB BEFORE the app boots.

    The lifespan hydrates the engine's per-city window from
    ``recent_readings_for_city``, so seeding here populates the rolling
    window the first poll cycle's z-score is judged against. Each
    reading is at ``base_time + i * interval_hours``.

    Tiny temperature drift (0.1°C per step) is added so the window's
    sample std is non-zero — we want the z-score path to engage, not
    the std-zero skip.
    """
    db = Database(path=db_path)
    await db.connect()
    try:
        out: list[Reading] = []
        for i in range(count):
            t_local = base_time + timedelta(hours=i * interval_hours)
            r = Reading(
                id=None,
                city=city,
                reading_time=t_local.strftime("%Y-%m-%dT%H:%M"),
                reading_time_utc=t_local.astimezone(UTC).isoformat(
                    timespec="seconds"
                ),
                fetched_at=utc_now_iso(),
                temperature_2m=base_temp + 0.1 * i,
                apparent_temperature=base_temp + 0.1 * i,
                precipitation=0.0,
                wind_speed_10m=wind_speed,
                weather_code=weather_code,
            )
            await db.insert_reading(r)
            out.append(r)
        return out
    finally:
        await db.close()


async def _wait_until(
    predicate: Callable[[], bool] | Callable[[], Any],
    *,
    timeout: float = 5.0,
    interval: float = 0.02,
    description: str = "",
) -> None:
    """Async polling helper: re-check ``predicate`` until True or timeout.

    The integration tests need to wait for "the poller has completed at
    least one cycle" without coupling to ``poll_interval_seconds``.
    Polling the DB row count is the right signal — ``insert_reading``
    is fully synchronous-from-our-perspective by the time the poller's
    cycle hands back control.

    Predicate may be sync or async; we await either way.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        await asyncio.sleep(interval)
    raise TimeoutError(
        f"Predicate not satisfied within {timeout}s"
        + (f": {description}" if description else "")
    )


def _settings_for(
    *,
    db_path: str,
    enable_poller: bool,
    poll_interval_seconds: int = 1,
    min_samples: int = 4,
    z_thresh: float = 2.5,
    cooldown_seconds: int = 10800,
    wind_thresh: float = 40.0,
) -> Settings:
    """Build a Settings tuned for fast deterministic integration tests.

    * ``poll_interval_seconds = 1``: the minimum the schema allows
      (``gt=0``, integer). With one cycle being enough for our tests,
      this just means "after the first cycle, wait 1s before the next"
      — which we cancel out by exiting the lifespan as soon as we see
      the expected DB rowcount.
    * ``min_samples = 4`` (default 6): so 5 seeded warm-ups easily
      clear the warm-up gate and the z-score detector engages.
    * ``cooldown_seconds = 10800`` (the production default) — keeps the
      cross-restart test honest.
    """
    return Settings(
        db_path=db_path,
        enable_poller=enable_poller,
        poll_interval_seconds=poll_interval_seconds,
        min_samples=min_samples,
        z_thresh=z_thresh,
        cooldown_seconds=cooldown_seconds,
        wind_thresh=wind_thresh,
    )


@pytest_asyncio.fixture
async def asgi_client_factory() -> AsyncIterator[
    Callable[[Settings], _AppHandle]
]:
    """Per-test factory that builds (app, ASGITransport, AsyncClient).

    Lifespan management is the test's responsibility (so we can
    deliberately START / STOP across two boots in the cross-restart
    test). The factory hands back a small handle so each test reads
    naturally.
    """
    handles: list[_AppHandle] = []

    def factory(settings: Settings) -> _AppHandle:
        app = create_app(settings)
        transport = ASGITransport(app=app)
        handle = _AppHandle(app=app, transport=transport, settings=settings)
        handles.append(handle)
        return handle

    yield factory

    # Defensive: any handle that opened a client without closing it (e.g.
    # because a test failed mid-way) gets cleaned up here so pytest
    # doesn't leak transports across tests.
    for h in handles:
        if h._client is not None and not h._client.is_closed:
            await h._client.aclose()


class _AppHandle:
    """Test handle pairing one app, its transport, and a lazy AsyncClient.

    Why this exists rather than a one-liner fixture:
    ``TestCooldownAcrossRealRestart`` boots TWO lifespans sequentially
    against the same DB file. Each boot needs its own app + transport +
    client, but they share settings. The handle keeps the bookkeeping
    explicit instead of buried in fixture composition.
    """

    def __init__(
        self,
        *,
        app: Any,
        transport: ASGITransport,
        settings: Settings,
    ) -> None:
        self.app = app
        self.transport = transport
        self.settings = settings
        self._client: AsyncClient | None = None

    async def open(self) -> AsyncClient:
        """Open the AsyncClient (does NOT enter the lifespan)."""
        self._client = AsyncClient(
            transport=self.transport, base_url="http://test"
        )
        return self._client

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("call .open() first")
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# 1. The centerpiece: end-to-end fire path
# ---------------------------------------------------------------------------


class TestEndToEndFirePath:
    """The receipt for the FIRE path.

    Symmetric to the M7 dress rehearsal's no-fire/dedup receipt
    (``new=3 -> duplicate=3``). This proves: with the full app booted
    via its lifespan, a real poller cycle pulling a (mocked) Open-Meteo
    response correctly drives the engine to detect, persist, and
    surface a notable event with a reason string carrying the actual
    numeric context.
    """

    @pytest.mark.asyncio
    async def test_temperature_anomaly_fires_with_correct_event_shape(
        self,
        tmp_path: Path,
        asgi_client_factory: Callable[[Settings], _AppHandle],
    ) -> None:
        db_path = str(tmp_path / "fire.db")

        # 1. Pre-seed Ottawa with 5 warm-up readings around 20°C. This
        #    populates the rolling window so the z-score path engages
        #    on the very first poll cycle (instead of sitting in
        #    warm-up fallback).
        seed_base = datetime(2026, 5, 28, 9, 0, tzinfo=UTC)
        await _seed_warmup_readings(
            db_path,
            city=OTTAWA.name,
            count=5,
            base_time=seed_base,
            base_temp=20.0,
        )

        # 2. Build the spike payload for Ottawa — temperature jump from
        #    ~20°C to 35°C. Mean of priors ≈ 20.2, std ≈ 0.16, so
        #    z ≈ 90 — astronomically over the 2.5σ threshold.
        spike_time_local = "2026-05-28T15:00"
        spike = _payload(
            city=(OTTAWA.latitude, OTTAWA.longitude),
            reading_time=spike_time_local,
            utc_offset_seconds=0,  # treat reading_time as UTC for clarity
            temperature_2m=35.0,
            apparent_temperature=35.0,
        )
        # Keep Toronto and Vancouver quiet — readings present but no
        # detector fires for them.
        toronto_calm = _payload(
            city=(TORONTO.latitude, TORONTO.longitude),
            reading_time="2026-05-28T15:00",
            utc_offset_seconds=0,
            temperature_2m=22.0,
        )
        vancouver_calm = _payload(
            city=(VANCOUVER.latitude, VANCOUVER.longitude),
            reading_time="2026-05-28T15:00",
            utc_offset_seconds=0,
            temperature_2m=18.0,
        )

        responder = _make_per_city_responder(
            {
                OTTAWA.latitude: spike,
                TORONTO.latitude: toronto_calm,
                VANCOUVER.latitude: vancouver_calm,
            }
        )

        settings = _settings_for(db_path=db_path, enable_poller=True)
        handle = asgi_client_factory(settings)

        with respx.mock(assert_all_called=False) as router:
            router.get(OPEN_METEO_URL).mock(side_effect=responder)

            client = await handle.open()
            try:
                async with handle.app.router.lifespan_context(handle.app):
                    # The lifespan started the poller. Wait until cycle
                    # 1 lands its 3 inserts on top of the 5 seeded rows
                    # (5 + 3 = 8). At that point detection has already
                    # run for Ottawa's spike.
                    db: Database = handle.app.state.db
                    await _wait_until(
                        lambda: db.count_readings(),
                        timeout=10.0,
                        description="cycle 1 to insert 3 readings",
                    )

                    async def _enough_readings() -> bool:
                        return (await db.count_readings()) >= 8

                    await _wait_until(
                        _enough_readings,
                        timeout=10.0,
                        description="≥ 8 readings (5 seeded + cycle 1's 3)",
                    )

                    # ------- the assertion block -----------------------
                    r = await client.get("/events")
                    assert r.status_code == 200, r.text
                    body = r.json()
                    assert set(body.keys()) == {"events"}, body.keys()

                    events = body["events"]
                    ottawa_anomalies = [
                        e for e in events
                        if e["city"] == OTTAWA.name
                        and e["event_type"] == "temperature_anomaly"
                    ]
                    assert len(ottawa_anomalies) == 1, (
                        f"expected exactly one temperature_anomaly for "
                        f"Ottawa, got {len(ottawa_anomalies)}: {events}"
                    )
                    event = ottawa_anomalies[0]

                    # Severity: with z > 4 the band must be 'high'.
                    assert event["severity"] == "high", event

                    # Reason carries the numbers: city name, the actual
                    # measurement, the σ figure, the trailing-N
                    # baseline, and the threshold. Loose substrings —
                    # we don't lock the exact format, just the data.
                    reason: str = event["reason"]
                    assert OTTAWA.name in reason
                    assert "35.0°C" in reason
                    assert "σ" in reason
                    assert "trailing-5" in reason or "trailing-" in reason
                    assert "threshold" in reason

                    # Context dict carries the structured numeric
                    # breakdown — the wire payload of the standout
                    # design choice.
                    ctx = event["context"]
                    assert ctx["method"] == "z_score"
                    assert ctx["value"] == pytest.approx(35.0, abs=1e-6)
                    assert ctx["window_size"] == 5
                    assert abs(ctx["z"]) >= 2.5  # cleared the threshold

                    # The event timestamp matches the spike payload's
                    # ``current.time`` (translated to UTC) — wires up
                    # the M3 reading_time_utc -> M5 detector ->
                    # M6 API path end-to-end.
                    expected_utc = reading_time_to_utc(
                        spike_time_local, 0
                    )
                    assert event["reading_time_utc"] == expected_utc

                    # /health agrees: 8 readings, 1 event.
                    h = await client.get("/health")
                    h_body = h.json()
                    assert h_body["readings_stored"] == 8
                    assert h_body["events_stored"] >= 1
            finally:
                await handle.close()


# ---------------------------------------------------------------------------
# 2. Multi-detector on one reading — INTEGRATION variant
# ---------------------------------------------------------------------------


class TestMultiDetectorOnOneReading:
    """Single payload trips multiple detectors at once.

    The unit test in ``test_detection_engine.py`` already pins this
    against the engine in isolation. This is the wired-system
    counterpart: the same guarantee holds when the poller, the
    OpenMeteoClient, and the API surface are in the path.
    """

    @pytest.mark.asyncio
    async def test_three_detectors_fire_for_one_reading_through_full_stack(
        self,
        tmp_path: Path,
        asgi_client_factory: Callable[[Settings], _AppHandle],
    ) -> None:
        db_path = str(tmp_path / "multi.db")

        # Warm-up readings: clear sky, calm wind, ~20°C, no precip.
        seed_base = datetime(2026, 5, 28, 9, 0, tzinfo=UTC)
        await _seed_warmup_readings(
            db_path,
            city=OTTAWA.name,
            count=5,
            base_time=seed_base,
            base_temp=20.0,
            wind_speed=10.0,
            weather_code=0,  # CLEAR tier
        )

        # The "perfect storm" payload. Trips at minimum:
        #   - temperature_anomaly  (35°C vs prior ~20°C, z huge)
        #   - wind_danger          (60 km/h ≥ 40 km/h threshold)
        #   - weather_code_transition (code 95 = THUNDERSTORM, escalates
        #     above the seeded CLEAR tier)
        # Apparent ≈ actual so feels_like_divergence stays quiet,
        # precipitation = 0 so heavy_precipitation / precipitation_onset
        # stay quiet. We're testing co-firing, not a kitchen sink.
        spike = _payload(
            city=(OTTAWA.latitude, OTTAWA.longitude),
            reading_time="2026-05-28T15:00",
            utc_offset_seconds=0,
            temperature_2m=35.0,
            apparent_temperature=35.0,
            wind_speed_10m=60.0,
            weather_code=95,
        )
        calm_t = _payload(
            city=(TORONTO.latitude, TORONTO.longitude),
            reading_time="2026-05-28T15:00",
            utc_offset_seconds=0,
        )
        calm_v = _payload(
            city=(VANCOUVER.latitude, VANCOUVER.longitude),
            reading_time="2026-05-28T15:00",
            utc_offset_seconds=0,
        )

        responder = _make_per_city_responder({
            OTTAWA.latitude: spike,
            TORONTO.latitude: calm_t,
            VANCOUVER.latitude: calm_v,
        })

        settings = _settings_for(db_path=db_path, enable_poller=True)
        handle = asgi_client_factory(settings)

        with respx.mock(assert_all_called=False) as router:
            router.get(OPEN_METEO_URL).mock(side_effect=responder)

            client = await handle.open()
            try:
                async with handle.app.router.lifespan_context(handle.app):
                    db: Database = handle.app.state.db

                    async def _enough_readings() -> bool:
                        return (await db.count_readings()) >= 8

                    await _wait_until(
                        _enough_readings,
                        timeout=10.0,
                        description="cycle 1 to add Ottawa's spike",
                    )

                    # Wait for at least 3 events to land. Each detector
                    # commits independently and the poller dispatches
                    # them within one ``on_new_reading`` call, but the
                    # event commits aren't bundled.
                    async def _enough_events() -> bool:
                        return (await db.count_events()) >= 3

                    await _wait_until(
                        _enough_events,
                        timeout=5.0,
                        description="3 events for Ottawa's spike",
                    )

                    r = await client.get("/events?city=Ottawa")
                    assert r.status_code == 200, r.text
                    events = r.json()["events"]

                    types = {e["event_type"] for e in events}
                    assert {
                        "temperature_anomaly",
                        "wind_danger",
                        "weather_code_transition",
                    } <= types, (
                        f"expected all three detectors to fire for "
                        f"Ottawa's spike; got {types}"
                    )

                    # All three events refer to the SAME underlying
                    # reading - same ``reading_time_utc``. The engine
                    # builds Event from the reading inside one
                    # ``on_new_reading`` call, so every detector that
                    # fires for that reading shares its UTC timestamp.
                    spike_utc = reading_time_to_utc("2026-05-28T15:00", 0)
                    spike_events = [
                        e for e in events
                        if e["reading_time_utc"] == spike_utc
                    ]
                    spike_types = {e["event_type"] for e in spike_events}
                    assert {
                        "temperature_anomaly",
                        "wind_danger",
                        "weather_code_transition",
                    } <= spike_types, spike_events
            finally:
                await handle.close()


# ---------------------------------------------------------------------------
# 3. URL query params — through the full stack, not just the unit test
# ---------------------------------------------------------------------------


class TestUrlQueryParamsAtIntegration:
    """The unit test in ``test_open_meteo.py`` already pins URL-param
    construction against an isolated client. This integration variant
    proves the same property holds when the URL is built inside the
    real lifespan, by the real poller's ``OpenMeteoClient`` instance
    (not a test-built one). Catches a refactor that tidies up the URL
    builder somewhere downstream and silently breaks the wire shape.
    """

    @pytest.mark.asyncio
    async def test_request_url_uses_decoded_query_params_through_full_stack(
        self,
        tmp_path: Path,
        asgi_client_factory: Callable[[Settings], _AppHandle],
    ) -> None:
        db_path = str(tmp_path / "url.db")

        responder = _make_per_city_responder({
            OTTAWA.latitude: _payload(
                city=(OTTAWA.latitude, OTTAWA.longitude),
                reading_time="2026-05-28T15:00",
                utc_offset_seconds=0,
            ),
            TORONTO.latitude: _payload(
                city=(TORONTO.latitude, TORONTO.longitude),
                reading_time="2026-05-28T15:00",
                utc_offset_seconds=0,
            ),
            VANCOUVER.latitude: _payload(
                city=(VANCOUVER.latitude, VANCOUVER.longitude),
                reading_time="2026-05-28T15:00",
                utc_offset_seconds=0,
            ),
        })

        settings = _settings_for(db_path=db_path, enable_poller=True)
        handle = asgi_client_factory(settings)

        with respx.mock(assert_all_called=False) as router:
            route = router.get(OPEN_METEO_URL).mock(side_effect=responder)

            await handle.open()
            try:
                async with handle.app.router.lifespan_context(handle.app):
                    db: Database = handle.app.state.db

                    async def _cycle_done() -> bool:
                        return (await db.count_readings()) >= 3

                    await _wait_until(
                        _cycle_done,
                        timeout=10.0,
                        description="cycle 1 hit all three cities",
                    )
            finally:
                await handle.close()

        # Inspect a captured request OUTSIDE the respx context (the
        # router still holds the call log). respx exposes the parsed
        # request including ``url.params`` already-decoded.
        assert route.calls.call_count >= 3, (
            f"expected respx to receive ≥3 requests (one per city), "
            f"got {route.calls.call_count}"
        )
        seen_lats: set[str] = set()
        for call in route.calls:
            req: httpx.Request = call.request

            # 1. The wire URL must NOT contain the HTML-entity bug.
            assert "&amp;" not in str(req.url), str(req.url)

            # 2. Query params must round-trip as decoded strings.
            qs = dict(req.url.params)
            assert qs["timezone"] == "auto"
            assert qs["wind_speed_unit"] == "kmh"
            assert "latitude" in qs
            assert "longitude" in qs

            # 3. ``current`` is the comma-joined list of fields — the
            #    string interpolation trap. Decoded, it must be a real
            #    comma-separated list, not a literal ``%2C`` blob.
            fields = qs["current"].split(",")
            assert set(fields) == {
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "wind_speed_10m",
                "weather_code",
            }, qs["current"]

            seen_lats.add(qs["latitude"])

        # All three cities were hit. (Decimal formatting may vary
        # slightly between httpx versions; we assert by parsed float
        # rather than exact string.)
        seen = {round(float(s), 4) for s in seen_lats}
        for c in CITIES:
            assert round(c.latitude, 4) in seen, (c.name, seen)


# ---------------------------------------------------------------------------
# 4. Cooldown survives a real restart — two real lifespans, same DB
# ---------------------------------------------------------------------------


class TestCooldownAcrossRealRestart:
    """Cooldown hydration across a real shutdown/start cycle.

    Unit tests in ``test_detection_engine.py`` and
    ``test_detection_cooldown.py`` cover the in-memory mechanics. This
    is the systems-level proof: boot, fire, shutdown, boot again
    against the same DB file, re-feed the same condition, and the
    second event MUST NOT fire. The container-restart "events that
    never stop firing" failure mode the spec calls out, with the
    actual lifecycle in the path.
    """

    @pytest.mark.asyncio
    async def test_wind_danger_does_not_re_fire_after_real_restart(
        self,
        tmp_path: Path,
        asgi_client_factory: Callable[[Settings], _AppHandle],
    ) -> None:
        db_path = str(tmp_path / "restart.db")

        # The two app boots share these settings (specifically the same
        # db_path). The cooldown is deliberately the production
        # default — we'd rather find a real bug than a tuned-low one.
        def _settings_factory() -> Settings:
            return _settings_for(
                db_path=db_path,
                enable_poller=True,
                # 3-hour cooldown matches production. The two boots
                # happen seconds apart, well inside the window.
                cooldown_seconds=10800,
            )

        # ---- BOOT #1: fire the wind_danger event ---------------------
        windy_ottawa_1 = _payload(
            city=(OTTAWA.latitude, OTTAWA.longitude),
            reading_time="2026-05-28T15:00",
            utc_offset_seconds=0,
            wind_speed_10m=60.0,  # ≥ 40 km/h threshold
        )
        calm = lambda c: _payload(  # noqa: E731 - tiny inline helper
            city=(c.latitude, c.longitude),
            reading_time="2026-05-28T15:00",
            utc_offset_seconds=0,
            wind_speed_10m=5.0,
        )
        responder_1 = _make_per_city_responder({
            OTTAWA.latitude: windy_ottawa_1,
            TORONTO.latitude: calm(TORONTO),
            VANCOUVER.latitude: calm(VANCOUVER),
        })

        handle_1 = asgi_client_factory(_settings_factory())
        with respx.mock(assert_all_called=False) as router:
            router.get(OPEN_METEO_URL).mock(side_effect=responder_1)

            await handle_1.open()
            try:
                async with handle_1.app.router.lifespan_context(handle_1.app):
                    db1: Database = handle_1.app.state.db

                    async def _events_present() -> bool:
                        return (await db1.count_events()) >= 1

                    await _wait_until(
                        _events_present,
                        timeout=10.0,
                        description="boot #1 fires wind_danger for Ottawa",
                    )
                    events_after_boot_1 = await db1.count_events()
                    readings_after_boot_1 = await db1.count_readings()
            finally:
                await handle_1.close()

        # Sanity for the test itself — boot 1 fired exactly one event.
        assert events_after_boot_1 == 1, events_after_boot_1
        # Each city contributed one reading on cycle 1.
        assert readings_after_boot_1 == 3, readings_after_boot_1

        # ---- BOOT #2: same DB, fresh app, same condition -------------
        # New reading_time so dedup doesn't block the insert (we WANT
        # the reading stored — what we DON'T want is the event firing
        # again).
        windy_ottawa_2 = _payload(
            city=(OTTAWA.latitude, OTTAWA.longitude),
            reading_time="2026-05-28T16:00",
            utc_offset_seconds=0,
            wind_speed_10m=65.0,  # still well over threshold
        )
        calm_later = lambda c: _payload(  # noqa: E731
            city=(c.latitude, c.longitude),
            reading_time="2026-05-28T16:00",
            utc_offset_seconds=0,
            wind_speed_10m=5.0,
        )
        responder_2 = _make_per_city_responder({
            OTTAWA.latitude: windy_ottawa_2,
            TORONTO.latitude: calm_later(TORONTO),
            VANCOUVER.latitude: calm_later(VANCOUVER),
        })

        handle_2 = asgi_client_factory(_settings_factory())
        with respx.mock(assert_all_called=False) as router:
            router.get(OPEN_METEO_URL).mock(side_effect=responder_2)

            await handle_2.open()
            try:
                async with handle_2.app.router.lifespan_context(handle_2.app):
                    db2: Database = handle_2.app.state.db

                    # Wait until cycle 1 of boot #2 lands its three new
                    # rows. Total readings: 3 (boot 1) + 3 (boot 2) = 6.
                    async def _new_cycle_landed() -> bool:
                        return (await db2.count_readings()) >= 6

                    await _wait_until(
                        _new_cycle_landed,
                        timeout=10.0,
                        description="boot #2 cycle inserted fresh readings",
                    )

                    # Give the engine a beat to process the on_new_reading
                    # hook (commit ordering: reading first, then any
                    # event from it). If a duplicate event were going to
                    # fire it would land here.
                    await asyncio.sleep(0.1)

                    events_after_boot_2 = await db2.count_events()
                    readings_after_boot_2 = await db2.count_readings()
            finally:
                await handle_2.close()

        # The headline assertion: no duplicate event. The cooldown
        # state, hydrated from the events table on boot #2's startup,
        # suppresses the would-be re-fire.
        assert events_after_boot_2 == 1, (
            f"cooldown failed to survive restart - boot #2 produced "
            f"{events_after_boot_2 - 1} duplicate event(s). Total "
            f"readings: {readings_after_boot_2}."
        )
        assert readings_after_boot_2 == 6, readings_after_boot_2


# ---------------------------------------------------------------------------
# 5. Per-test DB isolation — explicit pin for the cross-test bleed risk
# ---------------------------------------------------------------------------


class TestPerTestDatabaseIsolation:
    """An explicit pin for the M9 fixture-isolation invariant.

    With 240+ tests across 9 milestones we'd notice cross-test state
    pollution by now in aggregate ways (mysterious assertion failures
    only when the suite runs in some order). This test makes the
    invariant LOCAL: two parametrised runs land in separate
    ``tmp_path`` directories and each starts at zero rows / zero
    events. If a future fixture refactor swaps ``tmp_path`` for a
    module-scoped temp dir, this fails immediately and tells you
    where.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("run", [1, 2])
    async def test_each_run_gets_a_fresh_zero_row_database(
        self,
        tmp_path: Path,
        asgi_client_factory: Callable[[Settings], _AppHandle],
        run: int,
    ) -> None:
        # Distinct DB filename, but the parametrised tmp_path is what
        # actually guarantees isolation (it's a per-test directory).
        db_path = str(tmp_path / f"isolation-run-{run}.db")
        settings = _settings_for(db_path=db_path, enable_poller=False)
        handle = asgi_client_factory(settings)

        client = await handle.open()
        try:
            async with handle.app.router.lifespan_context(handle.app):
                r = await client.get("/health")
                assert r.status_code == 200
                body = r.json()
                # The headline guarantee: no row ever bleeds across runs.
                assert body["readings_stored"] == 0, (
                    f"run {run} expected a fresh DB but found "
                    f"{body['readings_stored']} pre-existing readings — "
                    f"fixture isolation broken."
                )
                assert body["events_stored"] == 0, (
                    f"run {run} expected a fresh DB but found "
                    f"{body['events_stored']} pre-existing events — "
                    f"fixture isolation broken."
                )
        finally:
            await handle.close()
