"""HTTP route handlers.

All handlers borrow the shared aiosqlite connection from
``request.app.state.db`` via the :func:`get_db` dependency — no
per-request connections.

Filter semantics (consistent across endpoints):

* ``city`` — optional exact-match filter. Unknown city → empty array,
  200 (it's an OPTIONAL FILTER, not a lookup).
* ``event_type`` / ``severity`` — optional exact-match filters on
  events; same "filter, not lookup" semantics.
* ``since`` — optional ISO-8601 datetime; rows with
  ``reading_time_utc >= since`` are returned. The handler normalises
  user input to the same format the DB column uses
  (``"YYYY-MM-DDTHH:MM:SS+00:00"``) so the WHERE clause is a clean
  string comparison.
* ``limit`` — required validated bound, ``1 ≤ limit ≤ 500``.
  ``limit=0`` or ``limit=600`` returns 422 via Pydantic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request

from watchagent.api.schemas import (
    EventOut,
    EventsResponse,
    HealthResponse,
    ReadingOut,
    ReadingsResponse,
)
from watchagent.storage import Database

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: shared DB from app state
# ---------------------------------------------------------------------------


def get_db(request: Request) -> Database:
    """Return the process-wide :class:`Database` from ``app.state``.

    The lifespan owns the connection; handlers borrow it. This is the
    pattern that prevents per-request connection sprawl, which would
    defeat WAL and surface as ``database is locked`` first.
    """
    db = getattr(request.app.state, "db", None)
    if db is None:
        # This should be impossible in production — the lifespan installs
        # ``app.state.db`` before any request can be served. If it does
        # happen (e.g. a test that constructs a route without lifespan
        # via ASGITransport), failing loudly is better than a silent None.
        msg = "app.state.db is not initialised; lifespan did not run"
        raise RuntimeError(msg)
    return db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_since(since: datetime | None) -> str | None:
    """Normalise a ``since`` datetime to the format the DB column uses.

    The column is ``"YYYY-MM-DDTHH:MM:SS+00:00"`` — produced via
    ``datetime.isoformat(timespec='seconds')`` on a UTC-aware datetime.
    User input may be:

    * naive (no tzinfo) → assumed UTC, the most defensible default for an
      API consumer who didn't say;
    * any timezone → converted to UTC.

    The result has the SAME shape as the DB column, so a string ``>=``
    comparison is correct.
    """
    if since is None:
        return None
    since = (
        since.replace(tzinfo=UTC) if since.tzinfo is None else since.astimezone(UTC)
    )
    return since.isoformat(timespec="seconds")


def _reading_to_out(r) -> ReadingOut:  # noqa: ANN001 — storage.Reading dataclass
    return ReadingOut(
        city=r.city,
        reading_time=r.reading_time,
        reading_time_utc=r.reading_time_utc,
        fetched_at=r.fetched_at,
        temperature_2m=r.temperature_2m,
        apparent_temperature=r.apparent_temperature,
        precipitation=r.precipitation,
        wind_speed_10m=r.wind_speed_10m,
        weather_code=r.weather_code,
    )


def _event_to_out(e) -> EventOut:  # noqa: ANN001 — storage.Event dataclass
    # storage.Event.id is None pre-insert; on the read path it's always set.
    return EventOut(
        id=e.id,
        city=e.city,
        event_type=e.event_type,
        reading_time=e.reading_time,
        reading_time_utc=e.reading_time_utc,
        detected_at=e.detected_at,
        severity=e.severity,
        reason=e.reason,
        context=e.context,
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(
    db: Annotated[Database, Depends(get_db)],
) -> HealthResponse:
    """Liveness + storage stats. Exact response shape:
    ``{"status": "ok", "readings_stored": N, "events_stored": M}``.

    No detector / poller probing here — those are implementation
    details. ``status: "ok"`` means "the API process is up and the DB
    is queryable"; if the DB connect failed, the lifespan would have
    refused to start.
    """
    return HealthResponse(
        readings_stored=await db.count_readings(),
        events_stored=await db.count_events(),
    )


# ---------------------------------------------------------------------------
# /readings
# ---------------------------------------------------------------------------


@router.get("/readings", response_model=ReadingsResponse)
async def list_readings(
    db: Annotated[Database, Depends(get_db)],
    city: Annotated[
        str | None, Query(description="Optional city filter (exact match).")
    ] = None,
    since: Annotated[
        datetime | None,
        Query(description="ISO-8601 timestamp; returns rows at or after."),
    ] = None,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=500,
            description="Max rows to return. 1 ≤ limit ≤ 500.",
        ),
    ] = 50,
) -> ReadingsResponse:
    rows = await db.select_readings(
        city=city,
        since=_normalise_since(since),
        limit=limit,
    )
    return ReadingsResponse(readings=[_reading_to_out(r) for r in rows])


# ---------------------------------------------------------------------------
# /events
# ---------------------------------------------------------------------------


@router.get("/events", response_model=EventsResponse)
async def list_events(
    db: Annotated[Database, Depends(get_db)],
    city: Annotated[
        str | None, Query(description="Optional city filter (exact match).")
    ] = None,
    type: Annotated[  # noqa: A002 — `type` is the public query-param name
        str | None,
        Query(description="Optional event_type filter (e.g. 'wind_danger')."),
    ] = None,
    severity: Annotated[
        Literal["low", "medium", "high"] | None,
        Query(description="Optional severity filter."),
    ] = None,
    since: Annotated[
        datetime | None,
        Query(description="ISO-8601 timestamp; returns events at or after."),
    ] = None,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=500,
            description="Max rows to return. 1 ≤ limit ≤ 500.",
        ),
    ] = 50,
) -> EventsResponse:
    rows = await db.select_events(
        city=city,
        event_type=type,
        severity=severity,
        since=_normalise_since(since),
        limit=limit,
    )
    return EventsResponse(events=[_event_to_out(e) for e in rows])
