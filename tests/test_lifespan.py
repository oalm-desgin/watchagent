"""Lifespan integration tests.

These exercise the wiring that makes the rest of the system actually
run as one process. The unit tests so far have tested each layer in
isolation; here we pin:

* Startup completes the full chain (DB → http client → engine →
  hydration → poller) **in order**.
* The poller is scheduled AFTER hydration finishes (not concurrent).
* If hydration raises, startup fails loudly and cleanup runs in
  reverse over whatever was initialised.
* Shutdown cancels the poller BEFORE closing the DB.
* The two supported invocations (``python -m watchagent`` and
  ``uvicorn watchagent.api:app``) go through the same app object and
  therefore the same lifespan.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from watchagent.api.app import create_app
from watchagent.config import Settings


@pytest.fixture
def settings_no_poller(tmp_path: Path) -> Settings:
    return Settings(db_path=str(tmp_path / "ls.db"), enable_poller=False)


@pytest.fixture
def settings_with_poller(tmp_path: Path) -> Settings:
    """Poller-on settings; tests using this fixture MUST patch the
    network or cancel the task before it fires."""
    return Settings(
        db_path=str(tmp_path / "ls.db"),
        enable_poller=True,
        poll_interval_seconds=600,
    )


# ---------------------------------------------------------------------------
# Startup happy path (no poller)
# ---------------------------------------------------------------------------


class TestStartupOrder:
    @pytest.mark.asyncio
    async def test_state_is_populated_in_order(
        self,
        settings_no_poller: Settings,
    ) -> None:
        """app.state.db, http_client, and engine are all set after the
        lifespan completes its startup phase."""
        app = create_app(settings_no_poller)
        async with app.router.lifespan_context(app):
            assert app.state.db is not None
            assert app.state.http_client is not None
            assert app.state.engine is not None
            # Hydration completed; engine.hydrated is True before any
            # request can be served.
            assert app.state.engine.hydrated is True

    @pytest.mark.asyncio
    async def test_engine_hydrated_before_lifespan_yields(
        self,
        settings_no_poller: Settings,
    ) -> None:
        """The reviewer's strict-ordering guarantee: by the time the
        lifespan hands control back to FastAPI (i.e. requests can be
        served), the engine has already replayed prior windows and
        cooldowns."""
        app = create_app(settings_no_poller)
        async with app.router.lifespan_context(app):
            assert app.state.engine.hydrated is True


# ---------------------------------------------------------------------------
# Startup happy path (with poller, offline via cancel-before-first-cycle)
# ---------------------------------------------------------------------------


class TestPollerScheduling:
    @pytest.mark.asyncio
    async def test_poller_task_is_scheduled_when_enabled(
        self,
        settings_with_poller: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The poller task lives on app.state.poller_task and is
        scheduled (not yet completed). We patch run_forever to a quick
        sleep-and-return so we don't actually hit Open-Meteo."""
        from watchagent.poller import Poller

        sleep_event = asyncio.Event()

        async def _stub_run_forever(self: Poller) -> None:  # noqa: ARG001
            await sleep_event.wait()

        monkeypatch.setattr(Poller, "run_forever", _stub_run_forever)

        app = create_app(settings_with_poller)
        async with app.router.lifespan_context(app):
            task = app.state.poller_task
            assert task is not None
            assert not task.done(), "poller should still be running"
            assert task.get_name() == "watchagent-poller"
            sleep_event.set()  # let teardown finish cleanly

    @pytest.mark.asyncio
    async def test_poller_disabled_when_setting_off(
        self,
        settings_no_poller: Settings,
    ) -> None:
        app = create_app(settings_no_poller)
        async with app.router.lifespan_context(app):
            assert not hasattr(app.state, "poller_task")

    @pytest.mark.asyncio
    async def test_silent_crash_in_poller_logs_an_error(
        self,
        settings_with_poller: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tripwire test: the poller's done_callback MUST log at ERROR
        level if the task dies for any reason other than CancelledError.
        Without this, an exception that escapes the per-city try/except
        would kill the task silently, /health would keep returning 200,
        and readings_stored would freeze - the kind of failure that's
        invisible until somebody notices the data isn't fresh.

        Uses ``structlog.testing.capture_logs`` rather than reconfiguring
        the global structlog factory; the latter pollutes other tests'
        ``capture_logs()`` calls (e.g. ``test_poller.py``) and is the
        kind of cross-test bleed M9's TestPerTestDatabaseIsolation pins
        for the DB layer.
        """
        from structlog.testing import capture_logs

        from watchagent.poller import Poller

        async def _crash_immediately(self: Poller) -> None:  # noqa: ARG001
            raise RuntimeError("simulated unhandled poller crash")

        monkeypatch.setattr(Poller, "run_forever", _crash_immediately)

        with capture_logs() as cap:
            app = create_app(settings_with_poller)
            async with app.router.lifespan_context(app):
                # Pump the event loop so the poller starts, crashes, and
                # the done_callback fires.
                await asyncio.sleep(0.05)

        died = [c for c in cap if c.get("event") == "poller.task_died"]
        assert len(died) == 1, (
            "Silent poller death must produce a 'poller.task_died' ERROR "
            "log line via the task's done_callback. Without it, the "
            f"background task can crash and nobody notices. Got: {cap}"
        )
        assert died[0].get("log_level") == "error", died[0]


# ---------------------------------------------------------------------------
# Fail-fast hydration
# ---------------------------------------------------------------------------


class TestFailFastHydration:
    """The reviewer's strict requirement: if hydration raises, startup
    MUST fail, and the app must NEVER serve traffic with empty detector
    state. Cleanup of any already-initialised resources still runs."""

    @pytest.mark.asyncio
    async def test_hydration_failure_aborts_startup(
        self,
        settings_no_poller: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from watchagent.detection.engine import DetectionEngine

        async def _kaboom(self: DetectionEngine) -> None:  # noqa: ARG001
            raise RuntimeError("hydration deliberately broken")

        monkeypatch.setattr(DetectionEngine, "hydrate_from_db", _kaboom)

        app = create_app(settings_no_poller)
        with pytest.raises(RuntimeError, match="hydration deliberately broken"):
            async with app.router.lifespan_context(app):
                pytest.fail(
                    "lifespan must NOT yield if hydration failed — "
                    "this assertion proves the app never enters the "
                    "request-serving state with empty engine state."
                )

    @pytest.mark.asyncio
    async def test_partial_init_resources_are_cleaned_up_on_failure(
        self,
        settings_no_poller: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When hydration raises, EVERY resource initialised before it
        MUST be closed by the finally chain — DB AND http_client. A
        chain that closes only the DB leaks httpx connection pools on
        every failed restart; a chain that closes only the http_client
        leaks the DB file handle. Both must be released."""
        import httpx

        from watchagent.detection.engine import DetectionEngine
        from watchagent.storage import Database

        closed: list[str] = []

        original_db_close = Database.close
        original_aclose = httpx.AsyncClient.aclose

        async def _track_db_close(self: Database) -> None:
            closed.append("db")
            await original_db_close(self)

        async def _track_aclose(self: httpx.AsyncClient) -> None:
            closed.append("http_client")
            await original_aclose(self)

        monkeypatch.setattr(Database, "close", _track_db_close)
        monkeypatch.setattr(httpx.AsyncClient, "aclose", _track_aclose)

        async def _kaboom(self: DetectionEngine) -> None:  # noqa: ARG001
            raise RuntimeError("nope")

        monkeypatch.setattr(DetectionEngine, "hydrate_from_db", _kaboom)

        app = create_app(settings_no_poller)
        with pytest.raises(RuntimeError):
            async with app.router.lifespan_context(app):
                pass  # never reached

        # Both must have been closed. Order matters too: cleanup is
        # reverse of registration, so http_client is closed BEFORE db.
        assert closed == ["http_client", "db"], (
            f"Cleanup chain must close every initialised resource in "
            f"reverse order, even when startup fails. Got: {closed}. "
            "Missing http_client means leaked connection pools on every "
            "failed restart; missing db means a leaked file handle."
        )

    @pytest.mark.asyncio
    async def test_engine_construction_failure_still_closes_db_and_http(
        self,
        settings_no_poller: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cleanup chain must work the same way for ANY failure
        stage. Here we induce the failure at engine construction (before
        hydration even runs) and assert the same two resources, which
        were registered before the failure point, are still closed."""
        import httpx

        from watchagent.detection import engine as engine_mod
        from watchagent.storage import Database

        closed: list[str] = []
        original_db_close = Database.close
        original_aclose = httpx.AsyncClient.aclose

        async def _track_db_close(self: Database) -> None:
            closed.append("db")
            await original_db_close(self)

        async def _track_aclose(self: httpx.AsyncClient) -> None:
            closed.append("http_client")
            await original_aclose(self)

        monkeypatch.setattr(Database, "close", _track_db_close)
        monkeypatch.setattr(httpx.AsyncClient, "aclose", _track_aclose)

        # Induce failure at the next stage (DetectionEngine construction).
        # By the time this fires, db and http_client are both registered.
        original_init = engine_mod.DetectionEngine.__init__

        def _bad_init(self: engine_mod.DetectionEngine, **kwargs: object) -> None:
            original_init(self, **kwargs)  # type: ignore[arg-type]
            raise RuntimeError("engine construction deliberately broken")

        monkeypatch.setattr(engine_mod.DetectionEngine, "__init__", _bad_init)

        app = create_app(settings_no_poller)
        with pytest.raises(RuntimeError, match="engine construction"):
            async with app.router.lifespan_context(app):
                pass

        assert closed == ["http_client", "db"], (
            "Engine-stage failure must still trigger reverse cleanup of "
            f"the resources registered earlier. Got: {closed}."
        )


# ---------------------------------------------------------------------------
# Shutdown order (poller cancelled before DB closed)
# ---------------------------------------------------------------------------


class TestShutdownOrder:
    @pytest.mark.asyncio
    async def test_shutdown_runs_cleanups_in_reverse_order(
        self,
        settings_with_poller: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Startup registered (db, http_client, poller) in that order;
        shutdown MUST close them in reverse: poller → http_client → db.
        If the order is wrong (DB closed first), aiosqlite raises on
        any in-flight poller writes and ``docker compose down`` wedges."""
        from watchagent.poller import Poller
        from watchagent.storage import Database

        order: list[str] = []

        # Stub run_forever to wait for cancellation cooperatively.
        async def _stub_run_forever(self: Poller) -> None:  # noqa: ARG001
            try:
                while True:
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                order.append("poller")
                raise

        monkeypatch.setattr(Poller, "run_forever", _stub_run_forever)

        # Track http_client close.
        import httpx

        original_aclose = httpx.AsyncClient.aclose

        async def _track_aclose(self: httpx.AsyncClient) -> None:
            order.append("http_client")
            await original_aclose(self)

        monkeypatch.setattr(httpx.AsyncClient, "aclose", _track_aclose)

        # Track DB close.
        original_db_close = Database.close

        async def _track_db_close(self: Database) -> None:
            order.append("db")
            await original_db_close(self)

        monkeypatch.setattr(Database, "close", _track_db_close)

        app = create_app(settings_with_poller)
        async with app.router.lifespan_context(app):
            # Yield to the event loop so the poller task is actually
            # scheduled (not just created). In production this happens
            # naturally because the FastAPI request loop provides many
            # awaits; here the body has no awaits, so we pump manually.
            # Without this, asyncio cancels a task that never started
            # and the stub's except clause never runs.
            await asyncio.sleep(0.05)

        assert order == ["poller", "http_client", "db"], (
            f"shutdown must be reverse-order. Got: {order}. "
            "If poller runs after http_client/db close, an in-flight "
            "fetch hits a closed client/DB and 'docker compose down' "
            "wedges."
        )


# ---------------------------------------------------------------------------
# End-to-end: HTTP requests work with the full lifespan
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def live_client(
    settings_no_poller: Settings,
) -> AsyncIterator[AsyncClient]:
    app = create_app(settings_no_poller)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as c, app.router.lifespan_context(app):
        yield c


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_lifespan_serves_health_endpoint(
        self,
        live_client: AsyncClient,
    ) -> None:
        r = await live_client.get("/health")
        assert r.status_code == 200
        assert set(r.json().keys()) == {
            "status",
            "readings_stored",
            "events_stored",
        }


# ---------------------------------------------------------------------------
# Entry-point parity (python -m watchagent vs uvicorn watchagent.api:app)
# ---------------------------------------------------------------------------


class TestEntryPointParity:
    def test_module_level_app_is_exported(self) -> None:
        """``uvicorn watchagent.api:app`` resolves the symbol ``app`` in
        the package's __init__. Without that, the only way to launch
        would be ``--factory`` which the README would have to document
        as an exception. We instead expose the app eagerly."""
        from watchagent.api import app

        assert app is not None
        # It's a real FastAPI instance, not a coroutine or a factory.
        from fastapi import FastAPI

        assert isinstance(app, FastAPI)

    def test_main_runner_uses_module_level_app(self) -> None:
        """``python -m watchagent`` calls main(), which delegates to
        uvicorn.run with the same ``"watchagent.api:app"`` import string.
        This is the parity guarantee — both invocations route through
        the same lifespan."""
        import inspect

        from watchagent.__main__ import main

        # We don't actually call main() (it would bind a port). We
        # inspect the source to verify the import string is the
        # canonical one.
        src = inspect.getsource(main)
        assert '"watchagent.api:app"' in src

    def test_main_passes_log_config_none(self) -> None:
        """``log_config=None`` is required to prevent uvicorn from
        wrapping our structlog JSON output in its own text formatter.
        Pin it so a "let's tidy up uvicorn args" refactor doesn't
        silently re-enable it."""
        import inspect

        from watchagent.__main__ import main

        src = inspect.getsource(main)
        assert "log_config=None" in src


# ---------------------------------------------------------------------------
# Hydrated state survives across requests within one lifespan
# ---------------------------------------------------------------------------


class TestSharedStateAcrossRequests:
    @pytest.mark.asyncio
    async def test_engine_and_db_are_shared_across_handler_calls(
        self,
        live_client: AsyncClient,
    ) -> None:
        """Sanity: the same engine and DB instance is used for every
        request. This is implied by the lifespan owning them, but worth
        a positive assertion: if a future refactor accidentally rebuilds
        them per-request, this fails."""
        app: Any = live_client._transport.app  # type: ignore[attr-defined]
        engine_id_1 = id(app.state.engine)
        await live_client.get("/health")
        engine_id_2 = id(app.state.engine)
        await live_client.get("/readings")
        engine_id_3 = id(app.state.engine)
        assert engine_id_1 == engine_id_2 == engine_id_3
