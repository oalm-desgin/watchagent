"""Entry point for ``python -m watchagent``.

Runs the FastAPI app via uvicorn, binding to ``API_HOST:API_PORT`` from
:mod:`watchagent.config`. The lifespan starts the DB + detection engine
+ poller; uvicorn handles signals and shutdown.

This is one of the two supported invocations:

* ``python -m watchagent``                       (this entry point)
* ``uvicorn watchagent.api:app --host ... --port ...``

Both go through the same module-level ``app`` and therefore the same
lifespan. The integration test in ``tests/test_lifespan.py`` pins
that contract.
"""

from __future__ import annotations

import uvicorn

from watchagent.config import Settings
from watchagent.logging_setup import configure_logging


def main() -> None:
    configure_logging()
    settings = Settings()

    # ``app`` is the module-level instance (from watchagent.api.__init__).
    # We pass it by import string so uvicorn handles the lifespan natively.
    # ``log_config=None`` keeps uvicorn from installing its own logging on
    # top of our structlog setup — otherwise the structured JSON we
    # produce gets wrapped in uvicorn's text formatter.
    uvicorn.run(
        "watchagent.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
        access_log=False,  # request logging is structlog's job, not uvicorn's
    )


if __name__ == "__main__":
    main()
