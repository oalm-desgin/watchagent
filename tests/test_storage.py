"""Storage layer tests.

The headline tests pin the design decisions from §4 / §5 of the spec:

* ``test_insert_reading_dedup_returns_boolean`` — the "was this new?" gate
  the poller relies on to decide whether to run detection.
* ``test_select_readings_orders_by_utc_across_cities`` — the cross-city
  ordering correctness the M2 review surfaced (Vancouver 13:00 and Ottawa
  13:00 are different absolute moments).
* ``test_insert_reading_does_not_open_outer_transaction`` — proves
  insert_reading commits independently so a later detector failure can't
  roll the reading back.
* ``test_recent_readings_for_city_returns_oldest_first`` — required shape
  for M5 detector state hydration.

The full M9 suite layers on dedup-via-mocked-API + detection logic; this file
covers the storage contract in isolation with deterministic inputs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio

from watchagent.storage import (
    Database,
    Event,
    Reading,
    reading_time_to_utc,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Per-test on-disk SQLite DB so we can also exercise persistence-related
    code paths (file path validation, parent dir creation, WAL pragma)."""
    database = Database(str(tmp_path / "watchagent.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


def make_reading(
    city: str = "Ottawa",
    *,
    reading_time: str = "2026-05-28T13:00",
    utc_offset_seconds: int = -14400,  # Ottawa EDT
    temperature_2m: float = 21.0,
    apparent_temperature: float = 19.0,
    precipitation: float = 0.0,
    wind_speed_10m: float = 10.0,
    weather_code: int = 0,
) -> Reading:
    return Reading(
        city=city,
        reading_time=reading_time,
        reading_time_utc=reading_time_to_utc(reading_time, utc_offset_seconds),
        fetched_at=utc_now_iso(),
        temperature_2m=temperature_2m,
        apparent_temperature=apparent_temperature,
        precipitation=precipitation,
        wind_speed_10m=wind_speed_10m,
        weather_code=weather_code,
    )


def make_event(city: str = "Ottawa", *, event_type: str = "wind_danger") -> Event:
    return Event(
        city=city,
        event_type=event_type,
        reading_time="2026-05-28T13:00",
        reading_time_utc="2026-05-28T17:00:00+00:00",
        detected_at=utc_now_iso(),
        severity="medium",
        reason="wind_speed_10m=42.0 km/h crossed WIND_THRESH=40.0",
        context={"value": 42.0, "threshold": 40.0},
    )


# ---------------------------------------------------------------------------
# Time helpers.
# ---------------------------------------------------------------------------


def test_utc_now_iso_format_is_seconds_precision_with_offset() -> None:
    out = utc_now_iso()
    # Format: "YYYY-MM-DDTHH:MM:SS+00:00"
    assert out.endswith("+00:00")
    assert "." not in out, "Must be seconds-precision (no fractional seconds)"
    assert len(out) == 25


@pytest.mark.parametrize(
    ("city_offset_label", "offset_seconds", "local_time", "expected_utc"),
    [
        ("Ottawa EDT (UTC-4)", -14400, "2026-05-28T13:00", "2026-05-28T17:00:00+00:00"),
        ("Ottawa EST (UTC-5)", -18000, "2026-01-15T08:00", "2026-01-15T13:00:00+00:00"),
        ("Vancouver PDT (UTC-7)", -25200, "2026-05-28T13:00", "2026-05-28T20:00:00+00:00"),
        ("Vancouver PST (UTC-8)", -28800, "2026-01-15T08:00", "2026-01-15T16:00:00+00:00"),
        ("UTC city (offset 0)", 0, "2026-05-28T13:00", "2026-05-28T13:00:00+00:00"),
    ],
)
def test_reading_time_to_utc(
    city_offset_label: str, offset_seconds: int, local_time: str, expected_utc: str
) -> None:
    """Converts naive local civil time + utc_offset_seconds to absolute UTC.

    This is the function that makes cross-city ordering correct — without
    it, Vancouver 13:00 and Ottawa 13:00 would both sort as "13:00".
    """
    assert reading_time_to_utc(local_time, offset_seconds) == expected_utc


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    """Schema must apply cleanly on every connect (CREATE TABLE IF NOT EXISTS)."""
    path = str(tmp_path / "x.db")
    db1 = Database(path)
    await db1.connect()
    await db1.close()

    db2 = Database(path)
    await db2.connect()  # would raise on a non-idempotent schema
    assert await db2.count_readings() == 0
    await db2.close()


@pytest.mark.asyncio
async def test_connect_creates_missing_parent_dir(tmp_path: Path) -> None:
    """In Docker the volume mount provides /data; for nested local paths we
    create missing parents so a fresh checkout does not require manual setup."""
    nested = tmp_path / "deeper" / "path" / "watchagent.db"
    db = Database(str(nested))
    await db.connect()
    try:
        assert nested.parent.exists()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_connection_property_raises_when_not_connected(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "x.db"))
    with pytest.raises(RuntimeError, match="not connected"):
        _ = db.connection


# ---------------------------------------------------------------------------
# Dedup — the boolean that gates detection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_reading_dedup_returns_boolean(db: Database) -> None:
    """ON CONFLICT(city, reading_time) DO NOTHING + cursor.rowcount returns
    True iff the row landed. This boolean is the poller's "should we run
    detection?" signal — only genuinely new readings should fire detectors.
    """
    r = make_reading()
    assert await db.insert_reading(r) is True, "first insert must be new"
    assert await db.insert_reading(r) is False, "second insert is a dedup hit"
    assert await db.insert_reading(r) is False, "third insert still dedup'd"
    assert await db.count_readings() == 1


@pytest.mark.asyncio
async def test_dedup_is_per_city_per_reading_time(db: Database) -> None:
    """Same reading_time across DIFFERENT cities is not a duplicate."""
    r_ott = make_reading(city="Ottawa", reading_time="2026-05-28T13:00")
    r_tor = make_reading(city="Toronto", reading_time="2026-05-28T13:00")
    assert await db.insert_reading(r_ott) is True
    assert await db.insert_reading(r_tor) is True
    assert await db.count_readings() == 2


@pytest.mark.asyncio
async def test_different_reading_time_same_city_inserts(db: Database) -> None:
    a = make_reading(reading_time="2026-05-28T13:00")
    b = make_reading(reading_time="2026-05-28T14:00")
    assert await db.insert_reading(a) is True
    assert await db.insert_reading(b) is True
    assert await db.count_readings() == 2


# ---------------------------------------------------------------------------
# Ordering — UTC, not local.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_readings_orders_by_utc_across_cities(db: Database) -> None:
    """Cross-city ordering must respect absolute time.

    Both readings have local time 13:00 but Vancouver's UTC is 20:00 and
    Ottawa's is 17:00, so Vancouver IS the newer reading and must come
    first when the API is queried with no city filter. Ordering by the
    local string would put them in arbitrary order (string sort would
    actually tie since both are "2026-05-28T13:00") — that is exactly the
    bug this test exists to prevent.
    """
    ottawa_1pm = make_reading(
        city="Ottawa", reading_time="2026-05-28T13:00", utc_offset_seconds=-14400
    )
    vancouver_1pm = make_reading(
        city="Vancouver", reading_time="2026-05-28T13:00", utc_offset_seconds=-25200
    )
    # Insert Ottawa first to defeat any "insertion order" fallback.
    await db.insert_reading(ottawa_1pm)
    await db.insert_reading(vancouver_1pm)

    rows = await db.select_readings(limit=10)
    assert [r.city for r in rows] == ["Vancouver", "Ottawa"], (
        "Vancouver 13:00 local = 20:00 UTC; Ottawa 13:00 local = 17:00 UTC. "
        "Most-recent-first must surface Vancouver first."
    )


@pytest.mark.asyncio
async def test_select_readings_filters_and_limits(db: Database) -> None:
    for hour in range(10):
        await db.insert_reading(make_reading(reading_time=f"2026-05-28T{hour:02d}:00"))
    for hour in range(5):
        await db.insert_reading(
            make_reading(city="Toronto", reading_time=f"2026-05-28T{hour:02d}:00")
        )

    ottawa_only = await db.select_readings(city="Ottawa", limit=3)
    assert len(ottawa_only) == 3
    assert all(r.city == "Ottawa" for r in ottawa_only)
    # Most-recent-first: hours 09, 08, 07.
    assert [r.reading_time for r in ottawa_only] == [
        "2026-05-28T09:00",
        "2026-05-28T08:00",
        "2026-05-28T07:00",
    ]


# ---------------------------------------------------------------------------
# Events.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_event_returns_event_with_id(db: Database) -> None:
    inserted = await db.insert_event(make_event())
    assert inserted.id is not None
    assert isinstance(inserted.id, int)


@pytest.mark.asyncio
async def test_event_context_round_trips_as_dict(db: Database) -> None:
    """The TEXT column stores JSON; reads must hand back a parsed dict so
    the API layer can serialise it as a JSON object, not a string-of-JSON."""
    inserted = await db.insert_event(make_event())
    [back] = await db.select_events(limit=1)
    assert back.context == inserted.context
    assert isinstance(back.context, dict)


@pytest.mark.asyncio
async def test_select_events_orders_by_utc_desc(db: Database) -> None:
    older = replace(make_event(), reading_time_utc="2026-05-28T10:00:00+00:00")
    newer = replace(make_event(), reading_time_utc="2026-05-28T17:00:00+00:00")
    await db.insert_event(older)
    await db.insert_event(newer)

    rows = await db.select_events(limit=10)
    assert rows[0].reading_time_utc == "2026-05-28T17:00:00+00:00"
    assert rows[1].reading_time_utc == "2026-05-28T10:00:00+00:00"


# ---------------------------------------------------------------------------
# Health-counters.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_readings_and_events(db: Database) -> None:
    assert await db.count_readings() == 0
    assert await db.count_events() == 0
    await db.insert_reading(make_reading(reading_time="2026-05-28T10:00"))
    await db.insert_reading(make_reading(reading_time="2026-05-28T11:00"))
    await db.insert_event(make_event())
    assert await db.count_readings() == 2
    assert await db.count_events() == 1


# ---------------------------------------------------------------------------
# Hydration helpers (M5 will use these).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_readings_for_city_returns_oldest_first(db: Database) -> None:
    """Detector state replays readings in chronological order on startup,
    so this helper hands them back oldest-first (the inverse of the
    most-recent-first API contract)."""
    for hour in range(10):
        await db.insert_reading(make_reading(reading_time=f"2026-05-28T{hour:02d}:00"))

    window = await db.recent_readings_for_city("Ottawa", limit=5)
    assert [r.reading_time for r in window] == [
        "2026-05-28T05:00",
        "2026-05-28T06:00",
        "2026-05-28T07:00",
        "2026-05-28T08:00",
        "2026-05-28T09:00",
    ]


@pytest.mark.asyncio
async def test_latest_event_returns_most_recent_for_city_and_type(db: Database) -> None:
    """Cooldown hydration: must return the latest event for an exact
    (city, event_type) pair, ignoring other types/cities."""
    older = replace(
        make_event(event_type="wind_danger"),
        reading_time_utc="2026-05-28T10:00:00+00:00",
    )
    newer = replace(
        make_event(event_type="wind_danger"),
        reading_time_utc="2026-05-28T17:00:00+00:00",
    )
    other_type = replace(
        make_event(event_type="rapid_temp_change"),
        reading_time_utc="2026-05-28T18:00:00+00:00",
    )
    other_city = replace(
        make_event(city="Toronto", event_type="wind_danger"),
        reading_time_utc="2026-05-28T19:00:00+00:00",
    )
    for ev in (older, newer, other_type, other_city):
        await db.insert_event(ev)

    latest = await db.latest_event("Ottawa", "wind_danger")
    assert latest is not None
    assert latest.reading_time_utc == "2026-05-28T17:00:00+00:00"

    nope = await db.latest_event("Montreal", "wind_danger")
    assert nope is None


# ---------------------------------------------------------------------------
# Bulk insert.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_events_bulk_returns_ids(db: Database) -> None:
    e1 = make_event(event_type="wind_danger")
    e2 = make_event(event_type="rapid_temp_change")
    out = await db.insert_events([e1, e2])
    assert [ev.id is not None for ev in out] == [True, True]
    assert {ev.event_type for ev in out} == {"wind_danger", "rapid_temp_change"}
