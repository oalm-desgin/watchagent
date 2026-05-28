"""Per-city rolling-window state.

Invariant the whole detection module relies on:

    The window holds the **prior** readings only — it does NOT contain the
    reading currently under evaluation.

Concretely, the engine calls each detector with ``evaluate(reading, state)``
and ONLY THEN calls ``state.add(reading)``. So when a detector consults
``state.window``, it sees the W (or fewer, during warm-up) readings that
came *before* the test point. This is exactly the "compute the z-score
from the prior window, test the current reading against that baseline"
discipline the brief calls out: the window the z-score sees and the point
it judges are disjoint, so a genuine spike isn't dampened by being folded
into its own mean.

The window is a ``deque(maxlen=W)``. Once it reaches W items, each
``add`` evicts the oldest reading, so the window is always "the most
recent W priors". No external trimming needed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from watchagent.storage import Reading


@dataclass(slots=True)
class CityState:
    """Rolling window of prior readings for a single city.

    Use :meth:`hydrate` once at startup with the most-recent-first list from
    :func:`Database.recent_readings_for_city` (which already returns
    oldest-first inside the window — see M3). After that the engine calls
    :meth:`add` after each evaluation, never before.
    """

    city: str
    capacity: int
    window: deque[Reading] = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("CityState.capacity must be >= 1")
        self.window = deque(maxlen=self.capacity)

    # -- mutation ---------------------------------------------------------

    def add(self, reading: Reading) -> None:
        """Append ``reading`` to the window. Oldest is evicted at capacity."""
        self.window.append(reading)

    def hydrate(self, readings: Iterable[Reading]) -> None:
        """Bulk-load priors at startup. Pass them in chronological order
        (oldest → newest); the deque preserves that order."""
        self.window.clear()
        for r in readings:
            self.window.append(r)

    # -- read accessors used by detectors --------------------------------

    @property
    def last(self) -> Reading | None:
        """The most-recent prior reading, or ``None`` if the window is empty.

        Used by detectors that need a single previous point to compute a
        delta (rate-of-change, precipitation onset, weather-code transition).
        """
        return self.window[-1] if self.window else None

    def temperatures(self) -> list[float]:
        """Snapshot of ``temperature_2m`` across the entire window."""
        return [r.temperature_2m for r in self.window]

    def __len__(self) -> int:
        return len(self.window)
