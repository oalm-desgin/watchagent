"""Structured JSON logging via structlog.

Every log line emitted anywhere in the service uses this configuration, so the
field schema is consistent: `event` (the message), `level`, `timestamp`, plus
contextual keys like `city`, `cycle_id`, `attempt`, `http_status`,
`reading_time`, `event_type`, `severity`. Downstream tools (or a human reading
docker logs) can grep / parse confidently.
"""

from __future__ import annotations

import logging
import sys

import structlog

from watchagent.config import settings

_configured = False


def configure_logging() -> None:
    """Idempotently install the structlog JSON pipeline and route stdlib through it."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger; auto-configures on first call."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()
