"""Poller tests.

Uses a real :class:`Database` (tmp-path SQLite) and a stub Open-Meteo client
that returns canned :class:`Reading` instances. This isolates the poller's
contract — gather() exception handling, dedup-vs-detection gating, per-city
isolation, summary correctness — from the HTTP retry policy that's already
covered in test_open_meteo.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio
from structlog.testing import capture_logs

from watchagent.cities import CITIES, City
from watchagent.poller import CycleSummary, Poller
from watchagent.storage import Database, Reading, reading_time_to_utc, utc_now_iso

# ---------------------------------------------------------------------------
# Stub Open-Meteo client.
# ---------------------------------------------------------------------------


class StubMeteo:
    """Mimics :class:`OpenMeteoClient` for poller-only tests.

    The real client's HTTP retry behaviour is covered separately; this stub
    lets us script per-city outcomes deterministically without spinning up
    respx / httpx for every poller scenario.
    """

    def __init__(
        self,
        per_city: dict[str, list[Reading | None | BaseException]],
    ) -> None:
        self.per_city = {k: list(v) for k, v in per_city.items()}
        self.calls: list[str] = []

    async def fetch_current(self, city: City) -> Reading | None:
        self.calls.append(city.name)
        queue = self.per_city.get(city.name, [])
        if not queue:
            raise AssertionError(f"StubMeteo: no scripted response for {city.name}")
        nxt = queue.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def make_reading(city: str, *, reading_time: str = "2026-05-28T13:00") -> Reading:
    return Reading(
        city=city,
        reading_time=reading_time,
        reading_time_utc=reading_time_to_utc(reading_time, -14400),
        fetched_at=utc_now_iso(),
        temperature_2m=20.0,
        apparent_temperature=18.0,
        precipitation=0.0,
        wind_speed_10m=10.0,
        weather_code=0,
    )


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(str(tmp_path / "poll.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


# ---------------------------------------------------------------------------
# Cycle summaries — new vs duplicate vs error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cycle_first_run_marks_every_city_new(db: Database) -> None:
    meteo = StubMeteo(
        per_city={
            "Ottawa": [make_reading("Ottawa")],
            "Toronto": [make_reading("Toronto")],
            "Vancouver": [make_reading("Vancouver")],
        }
    )
    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=CITIES,
        poll_interval_seconds=60,
    )
    summary = await poller.run_cycle()

    assert isinstance(summary, CycleSummary)
    assert summary.cities_polled == 3
    assert summary.new == 3
    assert summary.duplicate == 0
    assert summary.errors == 0
    assert await db.count_readings() == 3


@pytest.mark.asyncio
async def test_run_cycle_repolls_count_as_duplicates(db: Database) -> None:
    """Second cycle with the same readings -> 0 new, 3 duplicates.

    This is the contract the README will point at: "look at the logs —
    after the first cycle every poll is a duplicate, exactly because the
    upstream only refreshes hourly. Detection didn't run again, either."
    """
    readings = {
        "Ottawa": make_reading("Ottawa"),
        "Toronto": make_reading("Toronto"),
        "Vancouver": make_reading("Vancouver"),
    }
    meteo = StubMeteo(
        per_city={c: [r, r] for c, r in readings.items()},  # serve twice
    )
    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=CITIES,
        poll_interval_seconds=60,
    )

    first = await poller.run_cycle()
    second = await poller.run_cycle()

    assert (first.new, first.duplicate) == (3, 0)
    assert (second.new, second.duplicate) == (0, 3)
    assert await db.count_readings() == 3, "dedup must keep stored count flat"


@pytest.mark.asyncio
async def test_run_cycle_isolates_per_city_failures(db: Database) -> None:
    """Per-city errors do not affect other cities or break the cycle."""
    meteo = StubMeteo(
        per_city={
            "Ottawa": [make_reading("Ottawa")],
            "Toronto": [None],  # client returned None → error outcome
            "Vancouver": [make_reading("Vancouver")],
        }
    )
    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=CITIES,
        poll_interval_seconds=60,
    )
    summary = await poller.run_cycle()

    assert summary.new == 2
    assert summary.duplicate == 0
    assert summary.errors == 1
    assert await db.count_readings() == 2


@pytest.mark.asyncio
async def test_run_cycle_unhandled_exception_is_a_safety_net(db: Database) -> None:
    """The reviewer's M4 #3: gather(return_exceptions=True) does NOT handle
    exceptions — it stuffs them into the results list. The poller must
    iterate, detect Exception instances, log with city context, and
    convert each to an error outcome so one bad city cannot kill the cycle.

    We test this by forcing fetch_current to raise an unexpected error
    that _poll_city does NOT catch. (In the real client all errors are
    caught and surface as `None`; this test is the safety-net contract.)
    """
    meteo = StubMeteo(
        per_city={
            "Ottawa": [make_reading("Ottawa")],
            "Toronto": [RuntimeError("simulated bug in fetch_current")],
            "Vancouver": [make_reading("Vancouver")],
        }
    )
    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=CITIES,
        poll_interval_seconds=60,
    )

    with capture_logs() as cap:
        summary = await poller.run_cycle()

    # 2 new (Ottawa, Vancouver), 1 error (Toronto), no exception escapes.
    assert summary.new == 2
    assert summary.errors == 1
    assert summary.duplicate == 0

    unhandled_logs = [c for c in cap if c.get("event") == "poller.unhandled_exception"]
    assert len(unhandled_logs) == 1
    assert unhandled_logs[0]["city"] == "Toronto"
    assert unhandled_logs[0]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# on_new_reading hook — the M5 detection seam.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_new_reading_fires_only_for_new_readings(db: Database) -> None:
    """The hook is the detection gate: it must run for new readings and
    NOT run for duplicates (the rowcount-based dedup boolean drives this)."""
    r = make_reading("Ottawa")
    meteo = StubMeteo(per_city={"Ottawa": [r, r]})
    seen: list[str] = []

    async def hook(reading: Reading) -> None:
        seen.append(reading.reading_time)

    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=(CITIES[0],),
        poll_interval_seconds=60,
        on_new_reading=hook,
    )

    await poller.run_cycle()  # new
    await poller.run_cycle()  # duplicate

    assert seen == ["2026-05-28T13:00"], (
        "hook must fire exactly once — for the new reading, not the duplicate"
    )


@pytest.mark.asyncio
async def test_on_new_reading_failure_does_not_corrupt_summary(
    db: Database,
) -> None:
    """If detection raises, the reading still counts as 'new' (it WAS stored;
    only the post-storage hook failed). The cycle must not be killed."""
    meteo = StubMeteo(per_city={"Ottawa": [make_reading("Ottawa")]})

    async def broken_hook(reading: Reading) -> None:
        raise ValueError("boom")

    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=(CITIES[0],),
        poll_interval_seconds=60,
        on_new_reading=broken_hook,
    )

    summary = await poller.run_cycle()

    assert summary.new == 1
    assert summary.errors == 0  # the reading itself succeeded
    assert await db.count_readings() == 1


# ---------------------------------------------------------------------------
# Concurrency — cities polled in parallel.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cities_polled_independently(db: Database) -> None:
    """All three cities are touched in one cycle (gather, not sequential)."""
    meteo = StubMeteo(
        per_city={
            "Ottawa": [make_reading("Ottawa")],
            "Toronto": [make_reading("Toronto")],
            "Vancouver": [make_reading("Vancouver")],
        }
    )
    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=CITIES,
        poll_interval_seconds=60,
    )
    await poller.run_cycle()

    assert sorted(meteo.calls) == ["Ottawa", "Toronto", "Vancouver"]


# ---------------------------------------------------------------------------
# CycleSummary helper.
# ---------------------------------------------------------------------------


def test_cycle_summary_from_outcomes_counts_correctly() -> None:
    """Summary deserves its own unit test — it's what the README points at
    when claiming dedup is observable in the logs."""
    from watchagent.poller import PollOutcome

    outcomes = [
        PollOutcome(city="Ottawa", status="new", was_new=True),
        PollOutcome(city="Toronto", status="duplicate", was_new=False),
        PollOutcome(city="Vancouver", status="error", was_new=False),
    ]
    s = CycleSummary.from_outcomes("abc123", outcomes)
    assert s.cycle_id == "abc123"
    assert s.cities_polled == 3
    assert s.new == 1
    assert s.duplicate == 1
    assert s.errors == 1


# ---------------------------------------------------------------------------
# Cycle ID propagation through structlog contextvars.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cycle_id_appears_on_every_cycle_log(db: Database) -> None:
    """The cycle_id contextvar must propagate so all logs from one cycle
    correlate. This is what makes operator debugging possible — without it,
    a log line like 'partial_payload' has no link back to the cycle that
    triggered it."""
    meteo = StubMeteo(
        per_city={
            "Ottawa": [make_reading("Ottawa")],
            "Toronto": [make_reading("Toronto")],
            "Vancouver": [make_reading("Vancouver")],
        }
    )
    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=CITIES,
        poll_interval_seconds=60,
    )

    with capture_logs() as cap:
        summary = await poller.run_cycle()

    cycle_logs = [
        c for c in cap if c.get("event", "").startswith("poller.cycle.")
    ]
    assert len(cycle_logs) >= 2  # start + done
    cycle_ids = {c["cycle_id"] for c in cycle_logs}
    assert cycle_ids == {summary.cycle_id}


@pytest.mark.asyncio
async def test_unique_reading_times_per_city_across_cycles(db: Database) -> None:
    """Different reading_time per cycle -> all stored, all counted as new."""
    r1 = make_reading("Ottawa", reading_time="2026-05-28T13:00")
    r2 = replace(
        make_reading("Ottawa", reading_time="2026-05-28T14:00"),
        reading_time_utc=reading_time_to_utc("2026-05-28T14:00", -14400),
    )
    meteo = StubMeteo(per_city={"Ottawa": [r1, r2]})
    poller = Poller(
        meteo=meteo,  # type: ignore[arg-type]
        db=db,
        cities=(CITIES[0],),
        poll_interval_seconds=60,
    )
    s1 = await poller.run_cycle()
    s2 = await poller.run_cycle()
    assert (s1.new, s1.duplicate) == (1, 0)
    assert (s2.new, s2.duplicate) == (1, 0)
    assert await db.count_readings() == 2
