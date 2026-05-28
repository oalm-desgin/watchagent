"""Debouncer — the layer that decides whether a candidate event actually fires.

Detectors answer **"does the condition hold for this reading?"** (pure, no
time). This module answers **"given that condition holds, should we emit?"** —
applying edge detection (fire on transition into anomalous) and a per-(city,
event_type) cooldown override.

Spec rule, distilled:

    Fire on the transition INTO anomalous state. Suppress re-fire UNTIL
    the condition clears OR ``cooldown_seconds`` has elapsed (whichever
    comes first). After cooldown elapses on a *sustained* anomaly,
    re-fire and reset the cooldown — that way a 12-hour heatwave gets
    one event every 3 hours, not zero events after the first hour.

The clock is injectable. M5/M7 use ``datetime.now(UTC)``; tests use a
controllable callable. This is the same dependency-injection pattern
M4 used for ``sleep`` — and it's what makes the cooldown-survives-restart
test deterministic.

Hydration on startup:

* :meth:`hydrate` is called for each ``(city, event_type)`` that has a row
  in the ``events`` table. It seeds ``last_fire_at`` from
  :func:`Database.latest_event` AND sets ``in_anomalous`` based on whether
  that fire is still inside the cooldown window. Without that, the first
  reading after restart would re-fire as a "fresh edge transition" and
  emit a duplicate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass


log = structlog.get_logger(__name__)


def _default_clock() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class _DebounceState:
    """Internal per-(city, event_type) bookkeeping.

    ``in_anomalous`` is the edge-detection flag: ``True`` means the most
    recent reading we processed had ``condition_holds=True``. ``False``
    means we last saw a non-anomalous reading (or we just hydrated from a
    fire that's older than the cooldown window).
    """

    in_anomalous: bool = False
    last_fire_at: datetime | None = None


class Debouncer:
    """Per-(city, event_type) edge + cooldown gate.

    Public surface is two methods:

    * :meth:`hydrate` — seed state from the DB on startup.
    * :meth:`consume` — called by the engine for *every* (reading, detector)
      pair, with the detector's verdict on whether the condition held.
      Returns ``True`` if and only if this is the moment to emit an event.
    """

    def __init__(
        self,
        *,
        cooldown_seconds: int,
        clock: Callable[[], datetime] = _default_clock,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._states: dict[tuple[str, str], _DebounceState] = {}

    # -- hydration -------------------------------------------------------

    def hydrate(
        self,
        city: str,
        event_type: str,
        *,
        last_fire_at: datetime,
    ) -> None:
        """Seed cooldown state for ``(city, event_type)`` from the DB.

        ``last_fire_at`` is the ``detected_at`` of the most recent event of
        this type for this city. We compute ``in_anomalous`` ourselves based
        on whether that fire is still within the cooldown window — if it is,
        we want the next reading to hit the "sustained anomaly + cooldown
        not yet expired" suppression branch, not the "fresh edge" fire
        branch.
        """
        if last_fire_at.tzinfo is None:
            raise ValueError("last_fire_at must be timezone-aware (UTC)")

        elapsed = (self._clock() - last_fire_at).total_seconds()
        in_anomalous = elapsed < self._cooldown_seconds
        self._states[(city, event_type)] = _DebounceState(
            in_anomalous=in_anomalous,
            last_fire_at=last_fire_at,
        )
        log.debug(
            "debouncer.hydrate",
            city=city,
            event_type=event_type,
            last_fire_at=last_fire_at.isoformat(),
            elapsed_seconds=int(elapsed),
            in_anomalous=in_anomalous,
        )

    # -- main entry point ------------------------------------------------

    def consume(
        self,
        city: str,
        event_type: str,
        *,
        condition_holds: bool,
    ) -> bool:
        """Update state for ``(city, event_type)``; return whether to fire.

        Called once per (reading, detector) pair, ALWAYS — even when the
        detector returned ``None`` (``condition_holds=False``). That's how
        the in-anomalous flag clears: a non-anomalous reading drops it,
        re-arming the next 0→1 transition.

        **Cooldown semantics (the subtle decision worth pinning explicitly).**

        The brief says: "suppress re-fire UNTIL the condition clears OR
        ``cooldown_seconds`` elapses". Read literally, the OR has *two*
        escape hatches:

        1. **Observed clear → re-enter.** If we observe a reading with
           ``condition_holds=False`` and a later reading has
           ``condition_holds=True``, we treat the second one as a fresh
           edge and fire — even if the cooldown has not elapsed. The
           observed clear is taken at face value as the OR's escape hatch:
           by the brief's text, the suppression is over the moment we see
           the condition no longer holds. (Pinned by
           ``test_observed_clear_then_re_enter_within_cooldown_fires``.)

           The flapping concern this raises is real but bounded: the
           poller's hourly cadence is the floor against high-frequency
           noise. A 10-second flap can never reach the detector — the
           upstream sensor doesn't refresh that often, and our dedup
           gate would reject duplicates anyway. So "fire on every
           observed clear-and-re-enter" maps to "fire on every genuinely
           separate hourly weather event", which is the intended behavior.

        2. **Cooldown elapses on a sustained anomaly.** If
           ``condition_holds`` is True every time we ask, we suppress
           until ``cooldown_seconds`` has elapsed since the last fire,
           then re-fire and reset the cooldown anchor. This is what
           gives a 12-hour heatwave one event every 3 hours instead of
           zero events after the first. (Pinned by
           ``test_sustained_anomaly_refires_exactly_at_cooldown``.)

        **Restart edge case (hydration).** After a process restart we
        cannot replay the readings we missed, so we don't know whether
        the condition cleared while we were down. ``hydrate`` resolves
        this conservatively: if the previous fire is still inside the
        cooldown window, we set ``in_anomalous=True`` so the next True
        reading is treated as a continuation (suppressed), not a fresh
        edge. This errs toward fewer events; the alternative —
        re-firing every previously-anomalous condition immediately
        after restart — would be a flood every time we deploy.
        """
        key = (city, event_type)
        st = self._states.setdefault(key, _DebounceState())
        now = self._clock()

        if not condition_holds:
            # Condition cleared (or never held). Re-arm the edge detector;
            # leave last_fire_at untouched so a *future* fire still gets
            # the cooldown window vs. now (not vs. the previous fire that
            # already cleared its own suppression).
            st.in_anomalous = False
            return False

        # -- condition_holds is True from here ---------------------------

        if not st.in_anomalous:
            # Fresh edge: prior reading was clean, this one isn't. Fire,
            # mark in_anomalous, anchor the cooldown.
            st.in_anomalous = True
            st.last_fire_at = now
            return True

        # Sustained anomaly. Re-fire only if the cooldown window has
        # elapsed since the last fire. ``last_fire_at is None`` should
        # not happen in practice (we set it on the edge fire) but we
        # guard for it so a buggy test scenario doesn't deadlock the
        # detector.
        if st.last_fire_at is None:
            st.last_fire_at = now
            return True

        elapsed = (now - st.last_fire_at).total_seconds()
        if elapsed >= self._cooldown_seconds:
            st.last_fire_at = now
            return True

        return False

    # -- introspection ---------------------------------------------------

    def state_for(self, city: str, event_type: str) -> _DebounceState:
        """Read-only view of the internal state for ``(city, event_type)``.

        Tests use this; production code should not depend on it.
        """
        return self._states.setdefault((city, event_type), _DebounceState())
