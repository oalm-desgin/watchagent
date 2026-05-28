"""SQLite (WAL) persistence layer.

Design choices the README will reference:

* **Single connection, async-serialised.** ``aiosqlite`` runs SQLite calls in
  a worker thread; one ``Database`` instance owns one connection for the
  process lifetime and serialises every write through it. WAL mode (set on
  every connect) lets readers proceed concurrently with the single writer.
  The poller runs once per ``POLL_INTERVAL_SECONDS`` and writes ~3 rows per
  cycle, so contention is a non-issue at this scale.

* **Dedup is a DB-layer guarantee, not an app-layer hope.** ``readings`` has a
  ``UNIQUE(city, reading_time)`` constraint AND
  ``insert_reading`` uses ``INSERT … ON CONFLICT DO NOTHING`` and returns the
  ``cursor.rowcount`` boolean. That boolean is the clean "was this reading
  new?" signal the poller uses to gate detection — detection runs only on
  genuinely new rows, never on re-seen ones. Defence in depth: even if an
  app-layer check is bypassed, the DB constraint still rejects duplicates.

* **Ordering is by ``reading_time_utc``, not ``reading_time``.** With
  ``timezone=auto`` from Open-Meteo, ``reading_time`` is local civil time
  per city — Vancouver 13:00 and Ottawa 13:00 are different absolute moments.
  Sorting an across-cities query by the local string would mis-rank readings
  by ~3h. Every "most recent first" path (``/readings``, ``/events``,
  state-hydration windows) orders by ``reading_time_utc DESC`` instead.
  Indexes mirror that ordering so SQLite can scan the index without a sort.

* **Transaction separation (poller responsibility).** ``insert_reading`` and
  ``insert_event`` each commit independently. The poller commits the reading
  first, runs detection, then commits any events — so a detector bug never
  rolls back the raw reading (raw data is more valuable than derived events).

* **Timestamp format.** All UTC columns (``reading_time_utc``, ``fetched_at``,
  ``detected_at``) are produced via ``datetime.now(timezone.utc).isoformat(
  timespec="seconds")`` → e.g. ``"2026-05-28T04:00:00+00:00"``. SQLite TEXT
  sort agrees with chronological order under that single format.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import aiosqlite

from watchagent.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Row models — typed, in-memory representations of `readings` / `events` rows.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reading:
    city: str
    reading_time: str            # API-returned local civil time (e.g. "2026-05-28T00:00")
    reading_time_utc: str        # ISO-8601 with +00:00 (sortable)
    fetched_at: str              # ISO-8601 with +00:00 (sortable)
    temperature_2m: float
    apparent_temperature: float
    precipitation: float
    wind_speed_10m: float
    weather_code: int
    id: int | None = None        # set after insert; None pre-insert

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Self:
        return cls(
            id=row["id"],
            city=row["city"],
            reading_time=row["reading_time"],
            reading_time_utc=row["reading_time_utc"],
            fetched_at=row["fetched_at"],
            temperature_2m=row["temperature_2m"],
            apparent_temperature=row["apparent_temperature"],
            precipitation=row["precipitation"],
            wind_speed_10m=row["wind_speed_10m"],
            weather_code=row["weather_code"],
        )


@dataclass(frozen=True, slots=True)
class Event:
    city: str
    event_type: str
    reading_time: str
    reading_time_utc: str
    detected_at: str
    severity: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    id: int | None = None

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Self:
        return cls(
            id=row["id"],
            city=row["city"],
            event_type=row["event_type"],
            reading_time=row["reading_time"],
            reading_time_utc=row["reading_time_utc"],
            detected_at=row["detected_at"],
            severity=row["severity"],
            reason=row["reason"],
            context=json.loads(row["context"]) if row["context"] else {},
        )


# ---------------------------------------------------------------------------
# Time helpers — single source of truth for the UTC ISO-8601 format.
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return ``datetime.now(timezone.utc)`` formatted to seconds precision.

    Used for ``fetched_at`` and ``detected_at``. One format everywhere makes
    SQLite's lexical TEXT sort agree with chronological order, and tests
    parse without ambiguity.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def reading_time_to_utc(reading_time_local: str, utc_offset_seconds: int) -> str:
    """Convert Open-Meteo's naive local ``reading_time`` to ISO-8601 UTC.

    Open-Meteo returns ``current.time`` as a naive local string like
    ``"2026-05-28T13:00"`` (when ``timezone=auto``) plus a top-level
    ``utc_offset_seconds`` describing the city's offset.

    Math: parse the naive local string as if it were UTC to get a timestamp
    baseline, then subtract the city's UTC offset in seconds. For Ottawa
    (offset -14400), 13:00 local minus (-14400) → 17:00 UTC; for Vancouver
    (offset -25200), 13:00 local minus (-25200) → 20:00 UTC. Output matches
    :func:`utc_now_iso`'s format so SQLite TEXT sorts correctly.
    """
    naive_local = datetime.fromisoformat(reading_time_local)
    pseudo_utc_ts = naive_local.replace(tzinfo=UTC).timestamp()
    real_utc = datetime.fromtimestamp(pseudo_utc_ts - utc_offset_seconds, tz=UTC)
    return real_utc.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Schema.
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    city                 TEXT    NOT NULL,
    reading_time         TEXT    NOT NULL,
    reading_time_utc     TEXT    NOT NULL,
    fetched_at           TEXT    NOT NULL,
    temperature_2m       REAL    NOT NULL,
    apparent_temperature REAL    NOT NULL,
    precipitation        REAL    NOT NULL,
    wind_speed_10m       REAL    NOT NULL,
    weather_code         INTEGER NOT NULL,
    UNIQUE(city, reading_time)
);

-- Index on (city, reading_time_utc DESC) accelerates the per-city
-- "most recent first" read path and the M5 state-hydration window.
CREATE INDEX IF NOT EXISTS idx_readings_city_utc_desc
    ON readings(city, reading_time_utc DESC);

-- Index on reading_time_utc DESC accelerates the no-filter case;
-- this is what makes cross-city ordering correct (see module docstring).
CREATE INDEX IF NOT EXISTS idx_readings_utc_desc
    ON readings(reading_time_utc DESC);

CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    city             TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    reading_time     TEXT NOT NULL,
    reading_time_utc TEXT NOT NULL,
    detected_at      TEXT NOT NULL,
    severity         TEXT NOT NULL,
    reason           TEXT NOT NULL,
    context          TEXT NOT NULL  -- JSON-encoded dict
);

-- For cooldown hydration: "most recent event per (city, event_type)".
CREATE INDEX IF NOT EXISTS idx_events_city_type_utc_desc
    ON events(city, event_type, reading_time_utc DESC);

-- For /events without a city filter: cross-city most-recent-first.
CREATE INDEX IF NOT EXISTS idx_events_utc_desc
    ON events(reading_time_utc DESC);
"""

# Pragmas applied on every connect. Each one is load-bearing:
#
# * journal_mode = WAL  — mandated by the spec; lets readers proceed while
#   the single writer commits. Without it, every read blocks the writer.
# * synchronous = NORMAL — the documented correct pairing with WAL: durable
#   on a clean shutdown, slightly relaxed fsync semantics for performance.
#   FULL would be needlessly expensive; OFF would risk corruption.
# * busy_timeout = 5000 — the one people forget. With our single shared
#   connection in-process we are mostly safe from lock contention, but the
#   moment ANY second connection touches the DB (a test, the M10
#   data-analysis skill running while the poller writes, a future
#   sidecar reader) a missing busy_timeout turns a brief write-lock into
#   an immediate `database is locked` SQLITE_BUSY exception. With this
#   pragma SQLite waits up to 5s for the lock instead.
# * foreign_keys = ON — defensive; we don't have FKs yet but enabling now
#   means a future schema addition can rely on them.
# * temp_store = MEMORY — keeps SQLite's intermediate tables off disk;
#   small but free win on writes.
_PRAGMAS = (
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA busy_timeout = 5000;",
    "PRAGMA foreign_keys = ON;",
    "PRAGMA temp_store = MEMORY;",
)


# ---------------------------------------------------------------------------
# Database — connection wrapper + typed query helpers.
# ---------------------------------------------------------------------------


class Database:
    """Thin async wrapper around a single aiosqlite connection.

    Lifecycle: ``Database(path)`` constructs; ``await db.connect()`` opens the
    connection and applies pragmas + schema; ``await db.close()`` shuts it
    down. The FastAPI lifespan in M7 owns this lifecycle.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    # ----- lifecycle ----------------------------------------------------

    async def connect(self) -> None:
        if self._conn is not None:
            return
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        for pragma in _PRAGMAS:
            await self._conn.execute(pragma)
        await self._conn.commit()
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        log.info("storage.connected", db_path=self.path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            log.info("storage.closed", db_path=self.path)

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError(
                "Database not connected — call `await db.connect()` first."
            )
        return self._conn

    # ----- writes -------------------------------------------------------

    async def insert_reading(self, reading: Reading) -> bool:
        """Insert a reading and return True iff the row is genuinely new.

        Uses ``INSERT ... ON CONFLICT(city, reading_time) DO NOTHING`` so the
        DB enforces dedup. ``cursor.rowcount`` after a single-row insert is
        ``1`` when the row landed and ``0`` when the conflict suppressed it —
        that boolean is the clean "should we run detection?" gate.
        """
        sql = """
        INSERT INTO readings
            (city, reading_time, reading_time_utc, fetched_at,
             temperature_2m, apparent_temperature, precipitation,
             wind_speed_10m, weather_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(city, reading_time) DO NOTHING
        """
        params = (
            reading.city,
            reading.reading_time,
            reading.reading_time_utc,
            reading.fetched_at,
            reading.temperature_2m,
            reading.apparent_temperature,
            reading.precipitation,
            reading.wind_speed_10m,
            reading.weather_code,
        )
        async with self.connection.execute(sql, params) as cur:
            was_new = cur.rowcount == 1
        await self.connection.commit()
        return was_new

    async def insert_event(self, event: Event) -> Event:
        """Insert an event and return a new Event populated with its DB id.

        Each call commits independently so an event-insert failure cannot
        roll back the parent reading. Detection writes happen in a separate
        transaction from the reading write per §5 of the spec.
        """
        sql = """
        INSERT INTO events
            (city, event_type, reading_time, reading_time_utc,
             detected_at, severity, reason, context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            event.city,
            event.event_type,
            event.reading_time,
            event.reading_time_utc,
            event.detected_at,
            event.severity,
            event.reason,
            json.dumps(event.context, separators=(",", ":")),
        )
        async with self.connection.execute(sql, params) as cur:
            event_id = cur.lastrowid
        await self.connection.commit()
        return replace(event, id=event_id)

    # ----- reads --------------------------------------------------------

    async def select_readings(
        self, *, city: str | None = None, limit: int = 50
    ) -> list[Reading]:
        """Most-recent-first by ``reading_time_utc`` (cross-city correct)."""
        if city is not None:
            sql = (
                "SELECT * FROM readings WHERE city = ? "
                "ORDER BY reading_time_utc DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (city, limit)
        else:
            sql = "SELECT * FROM readings ORDER BY reading_time_utc DESC LIMIT ?"
            params = (limit,)
        async with self.connection.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [Reading.from_row(r) for r in rows]

    async def select_events(
        self, *, city: str | None = None, limit: int = 50
    ) -> list[Event]:
        """Most-recent-first by ``reading_time_utc`` (cross-city correct).

        We sort by ``reading_time_utc`` rather than ``detected_at`` so that
        events surface in the order the underlying weather happened. Under
        normal operation the two agree to within a poll cycle; under
        state-hydration replay (M5) ordering by ``detected_at`` would mix
        replayed and live events. ``reading_time_utc`` is the stable choice.
        """
        if city is not None:
            sql = (
                "SELECT * FROM events WHERE city = ? "
                "ORDER BY reading_time_utc DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (city, limit)
        else:
            sql = "SELECT * FROM events ORDER BY reading_time_utc DESC LIMIT ?"
            params = (limit,)
        async with self.connection.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [Event.from_row(r) for r in rows]

    async def count_readings(self) -> int:
        async with self.connection.execute("SELECT COUNT(*) FROM readings") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def count_events(self) -> int:
        async with self.connection.execute("SELECT COUNT(*) FROM events") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ----- hydration helpers (for M5 detector state on startup) ---------

    async def recent_readings_for_city(
        self, city: str, *, limit: int
    ) -> list[Reading]:
        """Return the most recent ``limit`` readings for ``city``, oldest-first.

        Detector state hydration: the rolling window is rebuilt by replaying
        these in chronological order, so we DESC-then-reverse to land them
        oldest-to-newest in memory.
        """
        sql = (
            "SELECT * FROM readings WHERE city = ? "
            "ORDER BY reading_time_utc DESC LIMIT ?"
        )
        async with self.connection.execute(sql, (city, limit)) as cur:
            rows = await cur.fetchall()
        return [Reading.from_row(r) for r in reversed(rows)]

    async def latest_event(self, city: str, event_type: str) -> Event | None:
        """Return the most recent event for ``(city, event_type)``, or None.

        Cooldown hydration: detector state on startup uses this to compute
        whether a given (city, event_type) is still inside its cooldown
        window. Without this, a container restart resets cooldowns and the
        next poll floods duplicate events.
        """
        sql = (
            "SELECT * FROM events WHERE city = ? AND event_type = ? "
            "ORDER BY reading_time_utc DESC LIMIT 1"
        )
        async with self.connection.execute(sql, (city, event_type)) as cur:
            row = await cur.fetchone()
        return Event.from_row(row) if row is not None else None

    # ----- bulk helpers -------------------------------------------------

    async def insert_events(self, events: Iterable[Event]) -> list[Event]:
        """Convenience wrapper: insert a sequence of events, return them with ids.

        Each row commits in its own transaction (delegates to insert_event)
        so a single event-insert failure does not roll back the others.
        """
        return [await self.insert_event(ev) for ev in events]
