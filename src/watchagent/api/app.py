"""FastAPI app factory + lifespan.

The lifespan owns the long-lived resources of the process — currently
just the :class:`Database` connection. M7 extends this to also build
the :class:`DetectionEngine`, hydrate detector state, and schedule the
poller, strictly in that order.

Splitting the app construction into a factory (``create_app``) rather
than a module-level singleton lets tests pass a fresh ``Settings`` with
a temp ``DB_PATH`` and a fresh app per test. The conventional
``app = create_app()`` is exposed at the package edge for production
``uvicorn`` invocations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from watchagent.api.routes import router
from watchagent.config import Settings
from watchagent.storage import Database

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app wired to a :class:`Database` via the lifespan.

    ``settings`` is optional: if omitted, falls back to the module-level
    instance from :mod:`watchagent.config`. Tests pass their own
    ``Settings`` with a temp ``DB_PATH`` to keep state isolated.
    """
    cfg = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("api.lifespan.startup", db_path=cfg.db_path)
        db = Database(path=cfg.db_path)
        await db.connect()
        app.state.db = db
        app.state.settings = cfg
        try:
            yield
        finally:
            log.info("api.lifespan.shutdown")
            await db.close()

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
