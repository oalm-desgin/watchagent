"""HTTP API surface.

Three endpoints, all read-only:

* ``GET /health``    — liveness + DB row counts
* ``GET /readings``  — most-recent-first weather readings, filterable
* ``GET /events``    — most-recent-first detected events, filterable

Design choices:

* **Single shared aiosqlite connection**, owned by the lifespan and
  exposed via ``app.state.db``. Handlers never open their own
  connections — opening per-request would multiply file handles, defeat
  the WAL setup, and make ``database is locked`` the first thing to
  surface under any load.
* **Pydantic response models** (in :mod:`.schemas`) so the response
  shape is locked at import time, not at runtime. Tests pin the exact
  key set with ``set(response.json().keys()) == {...}``; an accidental
  rename or extra field fails the contract test loudly instead of
  sailing past a "did it return 200" assertion.
* **Optional filters return empty arrays, validated bounds return 422.**
  Unknown ``city=Atlantis`` is a filter that matched zero rows
  (200 + ``{"readings": []}``), not a lookup that missed
  (would be 404). ``limit=0`` or ``limit=600`` violates the validated
  bound (1 ≤ limit ≤ 500) and returns 422 via Pydantic's automatic
  query-parameter validation.

The lifespan currently only manages the DB. M7 extends it to also
construct the :class:`DetectionEngine`, hydrate detector state, and
schedule the poller — strictly in that order, so the first poll cycle
never runs before hydration completes.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
