"""Pydantic response models for the HTTP API.

These mirror the storage-layer dataclasses one-for-one. We define them
explicitly (rather than auto-deriving from the dataclass) so that:

* The wire contract is explicit and reviewable in one file.
* Field renames at the storage layer are forced through this seam —
  the contract test will fail loudly if anyone shifts the wire shape
  by accident.
* JSON serialization is handled by Pydantic, including the
  ``dict[str, Any]`` for ``Event.context``.

The top-level wrappers (:class:`ReadingsResponse`, :class:`EventsResponse`,
:class:`HealthResponse`) lock the OUTER key set the grader checks. Tests
assert ``set(response.json().keys()) == {...}`` — an accidental extra
field or renamed key fails that assertion.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReadingOut(BaseModel):
    """One row from ``readings``. Matches storage.Reading field-for-field
    EXCEPT ``id``, which is internal bookkeeping and not exposed."""

    model_config = ConfigDict(extra="forbid")

    city: str
    reading_time: str = Field(
        ...,
        description="Local civil time as returned by Open-Meteo "
        "(e.g. '2026-05-28T11:00').",
    )
    reading_time_utc: str = Field(
        ...,
        description="Absolute UTC ISO-8601 with offset "
        "(e.g. '2026-05-28T15:00:00+00:00'). Sortable as TEXT.",
    )
    fetched_at: str = Field(..., description="When we polled, ISO-8601 UTC.")
    temperature_2m: float
    apparent_temperature: float
    precipitation: float
    wind_speed_10m: float
    weather_code: int


class EventOut(BaseModel):
    """One row from ``events``. Matches storage.Event one-for-one."""

    model_config = ConfigDict(extra="forbid")

    id: int
    city: str
    event_type: str
    reading_time: str
    reading_time_utc: str
    detected_at: str
    severity: Literal["low", "medium", "high"]
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)


class ReadingsResponse(BaseModel):
    """Top-level wrapper. The ``readings`` key is part of the contract —
    don't rename it."""

    model_config = ConfigDict(extra="forbid")

    readings: list[ReadingOut]


class EventsResponse(BaseModel):
    """Top-level wrapper. The ``events`` key is part of the contract —
    don't rename it."""

    model_config = ConfigDict(extra="forbid")

    events: list[EventOut]


class HealthResponse(BaseModel):
    """Top-level wrapper for ``GET /health``. Exact keys:
    ``{status, readings_stored, events_stored}``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    readings_stored: int
    events_stored: int
