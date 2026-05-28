"""The polling loop.

Owns the per-cycle orchestration:

1. Generate a ``cycle_id`` (UUID) and bind it to structlog contextvars so
   every log line in the cycle correlates.
2. Fan out across cities via ``asyncio.gather(*tasks, return_exceptions=True)``
   — slow Vancouver fetch must never delay Ottawa's storage. Detector
   state is per-city so there's no shared mutable state to race on.
3. Per city: fetch via :class:`OpenMeteoClient`, store via
   :meth:`Database.insert_reading`, and — if and only if the reading is
   genuinely new (the rowcount-based dedup boolean) — call the optional
   ``on_new_reading`` hook so detection runs.
4. Iterate gather's results explicitly, treating any ``BaseException`` as
   the safety net it actually is — ``return_exceptions=True`` does not
   handle exceptions, it just suppresses them and stuffs them into the
   results list. The classic bug is treating every result as a success.
5. Emit a structured per-cycle summary distinguishing
   ``new`` / ``duplicate`` / ``error`` so dedup is observable in the logs
   (which is what makes the README's "look at how the dedup works" story
   verifiable, not just claimed).

Lifecycle (M7 wires this up):
    * Lifespan startup creates the AsyncClient → OpenMeteoClient → Poller,
      then schedules ``poller_task = asyncio.create_task(poller.run_forever())``.
    * Lifespan shutdown does ``poller_task.cancel(); await poller_task``.
      :meth:`run_forever` propagates ``asyncio.CancelledError`` so shutdown
      is prompt instead of timing out.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Self

from structlog.contextvars import bound_contextvars

from watchagent.cities import City
from watchagent.logging_setup import get_logger
from watchagent.open_meteo import OpenMeteoClient
from watchagent.storage import Database, Reading

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Outcome / summary types — typed so the per-cycle summary log is well-formed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PollOutcome:
    city: str
    status: str  # "new" | "duplicate" | "error"
    was_new: bool
    reading: Reading | None = None


@dataclass(frozen=True, slots=True)
class CycleSummary:
    cycle_id: str
    cities_polled: int
    new: int
    duplicate: int
    errors: int

    @classmethod
    def from_outcomes(
        cls, cycle_id: str, outcomes: Sequence[PollOutcome]
    ) -> Self:
        return cls(
            cycle_id=cycle_id,
            cities_polled=len(outcomes),
            new=sum(1 for o in outcomes if o.status == "new"),
            duplicate=sum(1 for o in outcomes if o.status == "duplicate"),
            errors=sum(1 for o in outcomes if o.status == "error"),
        )


# ---------------------------------------------------------------------------
# Poller.
# ---------------------------------------------------------------------------


OnNewReading = Callable[[Reading], Awaitable[None]]


class Poller:
    """Periodically polls every configured city.

    The on_new_reading hook is the seam M5 plugs into: when present and the
    reading is new, the hook is awaited inside the per-city pipeline so
    detection runs in the same task that wrote the row.
    """

    def __init__(
        self,
        *,
        meteo: OpenMeteoClient,
        db: Database,
        cities: Sequence[City],
        poll_interval_seconds: int,
        on_new_reading: OnNewReading | None = None,
    ) -> None:
        self.meteo = meteo
        self.db = db
        self.cities = tuple(cities)
        self.poll_interval_seconds = poll_interval_seconds
        self.on_new_reading = on_new_reading

    # ----- lifecycle ---------------------------------------------------

    async def run_forever(self) -> None:
        """Run cycles every ``poll_interval_seconds`` until cancelled.

        Spec §5: the loop NEVER raises. A bug inside ``run_cycle`` is logged
        and the loop continues; only ``asyncio.CancelledError`` (from the
        lifespan shutdown) breaks out, and that is re-raised so the awaiting
        lifespan can complete promptly.
        """
        log.info(
            "poller.starting",
            poll_interval_seconds=self.poll_interval_seconds,
            cities=[c.name for c in self.cities],
        )
        try:
            while True:
                try:
                    await self.run_cycle()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Defence in depth: run_cycle already isolates per-city
                    # failures, so this should never trigger. If it does it's
                    # a bug in the cycle orchestrator itself; log and keep
                    # the loop alive so the next cycle still runs.
                    log.exception("poller.cycle_unhandled")
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            log.info("poller.shutdown")
            raise

    # ----- one cycle ---------------------------------------------------

    async def run_cycle(self) -> CycleSummary:
        """Fan out to every city, gather, summarise.

        Returns the :class:`CycleSummary` so tests (and future tooling) can
        inspect what happened beyond the side-effect log line.

        ``cycle_id`` is bound to structlog contextvars (so every nested log
        from :meth:`_poll_city` and :class:`OpenMeteoClient` carries it
        implicitly via the JSON pipeline's merge_contextvars processor) AND
        passed explicitly to the cycle.start / cycle.done summary lines so
        those lines are self-contained and testable in isolation.
        """
        cycle_id = uuid.uuid4().hex
        with bound_contextvars(cycle_id=cycle_id):
            log.info(
                "poller.cycle.start",
                cycle_id=cycle_id,
                cities=[c.name for c in self.cities],
            )

            tasks = [self._poll_city(city) for city in self.cities]
            # `return_exceptions=True` does NOT handle exceptions — it just
            # stops them from cancelling the gather and stuffs them into the
            # results list. We MUST iterate and detect Exception instances.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            outcomes: list[PollOutcome] = []
            for city, result in zip(self.cities, results, strict=True):
                if isinstance(result, BaseException):
                    log.error(
                        "poller.unhandled_exception",
                        cycle_id=cycle_id,
                        city=city.name,
                        error_type=type(result).__name__,
                        error=str(result),
                        exc_info=(type(result), result, result.__traceback__),
                    )
                    outcomes.append(
                        PollOutcome(
                            city=city.name,
                            status="error",
                            was_new=False,
                        )
                    )
                else:
                    outcomes.append(result)

            summary = CycleSummary.from_outcomes(cycle_id, outcomes)
            log.info(
                "poller.cycle.done",
                cycle_id=cycle_id,
                cities_polled=summary.cities_polled,
                new=summary.new,
                duplicate=summary.duplicate,
                errors=summary.errors,
            )
            return summary

    # ----- per city ----------------------------------------------------

    async def _poll_city(self, city: City) -> PollOutcome:
        """Fetch + dedup + store + (optionally) detect, all under a city
        contextvars binding so every log line carries the city."""
        with bound_contextvars(city=city.name):
            reading = await self.meteo.fetch_current(city)
            if reading is None:
                # Client already logged the underlying failure cause; we just
                # mark this city's outcome.
                return PollOutcome(
                    city=city.name, status="error", was_new=False
                )

            was_new = await self.db.insert_reading(reading)

            if was_new and self.on_new_reading is not None:
                # Detection failures must not corrupt this cycle's summary —
                # the reading is already committed by the time we get here.
                try:
                    await self.on_new_reading(reading)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(
                        "poller.on_new_reading_failed",
                        reading_time=reading.reading_time,
                    )

            return PollOutcome(
                city=city.name,
                status="new" if was_new else "duplicate",
                was_new=was_new,
                reading=reading,
            )
