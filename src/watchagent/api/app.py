"""FastAPI app factory + lifespan.

The lifespan owns every long-lived resource in the process: the DB
connection, the shared httpx client, the detection engine, and the
background poller task. Ordering matters in BOTH directions:

**Startup** (each stage MUST complete before the next):

1. Connect the DB (schema + pragmas applied).
2. Open the shared ``httpx.AsyncClient``.
3. Build the detection engine (debouncer + per-city windows + detectors).
4. Hydrate the engine from the DB — windows from
   ``recent_readings_for_city(W)``, cooldowns from ``latest_event(...)``.
   **Fail-fast**: if hydration raises, the whole startup is aborted and
   the cleanup chain runs in reverse. The app NEVER serves traffic with
   empty detector state.
5. Schedule the poller task. Only after hydration completes, so the
   first poll cycle sees the correct prior-window state.

**Shutdown** (reverse order, defensive: each step in its own try/except
so one cleanup failure doesn't skip the rest):

1. Cancel the poller task and await it — no fetch can be in flight when
   we close downstream resources.
2. Close the shared httpx client.
3. Close the DB.

If shutdown skipped step 1 and closed the DB while a fetch was mid-cycle,
``aiosqlite`` would raise on the next insert and either wedge
``docker compose down`` or leave a WAL file dirty.

The factory pattern (``create_app(settings)``) lets tests inject a
per-test ``DB_PATH`` and ``enable_poller=False``. The module-level
``app`` (in :mod:`watchagent.api.__init__`) is what ``uvicorn`` binds.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI

from watchagent.api.routes import router
from watchagent.cities import CITIES
from watchagent.config import Settings
from watchagent.detection import Debouncer, DetectionEngine
from watchagent.detection.engine import build_default_detectors
from watchagent.open_meteo import OpenMeteoClient
from watchagent.poller import Poller
from watchagent.storage import Database

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with the full lifespan wired up.

    Args:
        settings: Configuration. Defaults to a fresh ``Settings()`` (env-driven).
    """
    cfg = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Registered cleanups, popped in reverse order. Each entry is
        # (name, async_callable). We register only AFTER the resource is
        # successfully initialised, so a partial-startup failure cleans
        # up exactly what was created and nothing more.
        cleanups: list[tuple[str, Callable[[], Awaitable[None]]]] = []

        try:
            log.info(
                "api.lifespan.startup.begin",
                db_path=cfg.db_path,
                enable_poller=cfg.enable_poller,
            )

            # --- Stage 1: DB ---------------------------------------------
            db = Database(path=cfg.db_path)
            await db.connect()
            cleanups.append(("db", db.close))
            app.state.db = db
            app.state.settings = cfg
            log.info("api.lifespan.db_ready")

            # --- Stage 2: shared httpx client ----------------------------
            # One client for the process lifetime: connection pooling, TLS
            # reuse. Open-Meteo's timeout config is per-request via the
            # ``OpenMeteoClient`` wrapper.
            http_client = httpx.AsyncClient(
                timeout=cfg.http_timeout_seconds,
            )
            cleanups.append(("http_client", http_client.aclose))
            app.state.http_client = http_client
            log.info("api.lifespan.http_client_ready")

            # --- Stage 3: detection engine -------------------------------
            debouncer = Debouncer(cooldown_seconds=cfg.cooldown_seconds)
            engine = DetectionEngine(
                db=db,
                cities=CITIES,
                detectors=build_default_detectors(cfg),
                debouncer=debouncer,
                window_capacity=cfg.w,
            )

            # --- Stage 4: HYDRATE (fail-fast) ----------------------------
            # Any exception here aborts the startup. The cleanup chain
            # runs and the app refuses to come up rather than serve with
            # empty detector state.
            log.info("api.lifespan.hydration.begin")
            await engine.hydrate_from_db()
            app.state.engine = engine
            log.info(
                "api.lifespan.hydration.done",
                cities=len(CITIES),
                detectors=len(engine.detectors),
            )

            # --- Stage 5: poller task ------------------------------------
            if cfg.enable_poller:
                meteo = OpenMeteoClient(
                    client=http_client,
                    max_retries=cfg.max_retries,
                    backoff_base_seconds=cfg.retry_backoff_base_seconds,
                )
                poller = Poller(
                    meteo=meteo,
                    db=db,
                    cities=CITIES,
                    poll_interval_seconds=cfg.poll_interval_seconds,
                    on_new_reading=engine.on_new_reading,
                )
                poller_task = asyncio.create_task(
                    poller.run_forever(),
                    name="watchagent-poller",
                )

                # Silent-crash tripwire. Without this, an exception that
                # somehow escaped the poller's per-city try/except would
                # kill the task, the GC would eventually warn "Task
                # exception was never retrieved" to stderr (easy to miss
                # in JSON-log streams), /health would keep returning 200,
                # and readings_stored would silently stop growing. With
                # the callback, an unexpected exception is a single
                # ERROR-level structured log line, immediately.
                #
                # CancelledError is the normal shutdown path - we filter
                # it out so a clean stop doesn't generate a false alarm.
                def _on_poller_done(task: asyncio.Task[None]) -> None:
                    if task.cancelled():
                        return
                    exc = task.exception()
                    if exc is not None:
                        log.error(
                            "poller.task_died",
                            exc_info=exc,
                            note=(
                                "background poller crashed; the API will "
                                "keep responding but no new readings will "
                                "be polled until the process is restarted"
                            ),
                        )

                poller_task.add_done_callback(_on_poller_done)
                app.state.poller = poller
                app.state.poller_task = poller_task

                async def _stop_poller() -> None:
                    """Cooperative cancel: signal cancel, await the
                    CancelledError so any in-flight fetch finishes
                    cleanly. Re-raising any other exception lets the
                    finally chain log it via its own try/except."""
                    poller_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await poller_task

                cleanups.append(("poller", _stop_poller))
                log.info(
                    "api.lifespan.poller_started",
                    poll_interval_seconds=cfg.poll_interval_seconds,
                )
            else:
                log.info("api.lifespan.poller_disabled_by_settings")

            log.info("api.lifespan.startup.complete")
            yield

        except Exception:
            # Hydration / startup failure path. We log loudly and re-raise
            # so FastAPI / uvicorn treats startup as failed. The finally
            # block still runs to tear down whatever WAS initialised.
            log.exception("api.lifespan.startup.failed")
            raise
        finally:
            log.info("api.lifespan.shutdown.begin")
            # Reverse order. Each cleanup gets its own try/except so a
            # bad cleanup doesn't skip the rest.
            for name, cleanup in reversed(cleanups):
                try:
                    await cleanup()
                    log.info("api.lifespan.cleanup.ok", resource=name)
                except Exception:
                    log.exception(
                        "api.lifespan.cleanup.failed",
                        resource=name,
                    )
            log.info("api.lifespan.shutdown.complete")

    app = FastAPI(
        title="WatchAgent",
        description=(
            "Live weather monitor for Ottawa, Toronto, and Vancouver. "
            "Polls Open-Meteo every cycle, persists readings, runs a "
            "detector pipeline, and surfaces notable events."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
