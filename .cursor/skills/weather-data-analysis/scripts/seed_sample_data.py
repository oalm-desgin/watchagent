#!/usr/bin/env python3
"""Generate a realistic WatchAgent demo database for the analysis skill.

This is OPTIONAL — in normal use the poller fills the DB from live
Open-Meteo data. This helper exists so the analysis skill can be
demonstrated (and its example output regenerated) without waiting
hours for real weather to do something notable.

It is deliberately NOT a shortcut around the detection logic: each
synthetic reading is inserted via the real ``Database`` and then fed
through the real ``DetectionEngine.on_new_reading``. The events it
produces come from the actual detectors and the actual ``Debouncer``,
so ``event-breakdown`` shows genuine detector output, not fabricated
rows.

Run (from the repo root, with the package importable):

    PYTHONPATH=src python .cursor/skills/weather-data-analysis/scripts/seed_sample_data.py --db demo.db

On Windows PowerShell:

    $env:PYTHONPATH="src"; python .cursor/skills/weather-data-analysis/scripts/seed_sample_data.py --db demo.db

It writes ~48 hourly readings per city across Ottawa, Toronto, and
Vancouver, with a few injected notable conditions (an Ottawa heat
spike, a Toronto wind + thunderstorm escalation, a Vancouver rain
onset, a wind-chill divergence) so every analysis mode has content.
"""

from __future__ import annotations

import argparse
import asyncio
import math
from datetime import UTC, datetime, timedelta

from watchagent.cities import CITIES
from watchagent.config import Settings
from watchagent.detection import Debouncer, DetectionEngine
from watchagent.detection.engine import build_default_detectors
from watchagent.storage import Database, Reading, reading_time_to_utc, utc_now_iso

# Realistic local UTC offsets (seconds) for late May: EDT for ON, PDT for BC.
OFFSETS = {"Ottawa": -14400, "Toronto": -14400, "Vancouver": -25200}
BASELINE_C = {"Ottawa": 20.0, "Toronto": 22.0, "Vancouver": 15.0}
HOURS = 48


def _reading(
    *,
    city: str,
    local_dt: datetime,
    temp: float,
    apparent: float,
    precip: float,
    wind: float,
    code: int,
) -> Reading:
    local_str = local_dt.strftime("%Y-%m-%dT%H:%M")
    return Reading(
        id=None,
        city=city,
        reading_time=local_str,
        reading_time_utc=reading_time_to_utc(local_str, OFFSETS[city]),
        fetched_at=utc_now_iso(),
        temperature_2m=round(temp, 1),
        apparent_temperature=round(apparent, 1),
        precipitation=round(precip, 2),
        wind_speed_10m=round(wind, 1),
        weather_code=code,
    )


def _series(city: str, start_local: datetime) -> list[Reading]:
    """Build one city's hourly series with a few injected notable hours."""
    base = BASELINE_C[city]
    out: list[Reading] = []
    for h in range(HOURS):
        local_dt = start_local + timedelta(hours=h)
        # Mild diurnal swing so the z-score baseline has a realistic,
        # non-zero spread (peaks mid-afternoon).
        diurnal = 5.0 * math.sin((h % 24 - 9) / 24 * 2 * math.pi)
        temp = base + diurnal
        apparent = temp - 1.0
        precip = 0.0
        wind = 12.0
        code = 1  # mainly clear

        # --- injected notable conditions -------------------------------
        if city == "Ottawa" and h == 30:
            # Heat spike: trips temperature_anomaly (z) AND rapid_temp_change.
            temp = 35.0
            apparent = 37.0
            code = 2
        if city == "Toronto" and h == 20:
            # Severe storm: wind_danger + weather_code escalation to 95.
            wind = 65.0
            code = 95
            precip = 6.0  # also moderate heavy_precipitation
        if city == "Vancouver" and h == 10:
            # Rain onset (dry -> wet): precipitation_onset + heavy_precip.
            precip = 5.0
            code = 63
        if city == "Vancouver" and h == 33:
            # Wind-chill divergence: feels much colder than the air.
            apparent = temp - 8.0
            wind = 38.0

        out.append(
            _reading(
                city=city,
                local_dt=local_dt,
                temp=temp,
                apparent=apparent,
                precip=precip,
                wind=wind,
                code=code,
            )
        )
    return out


async def seed(db_path: str) -> None:
    db = Database(path=db_path)
    await db.connect()
    try:
        cfg = Settings(db_path=db_path, enable_poller=False)
        # The engine holds one CityState per known city, so it must be
        # built with the real CITIES tuple — that's what on_new_reading
        # looks up by reading.city.
        engine = DetectionEngine(
            db=db,
            cities=CITIES,
            detectors=build_default_detectors(cfg),
            debouncer=Debouncer(cooldown_seconds=cfg.cooldown_seconds),
            window_capacity=cfg.w,
        )

        start_local = datetime(2026, 5, 26, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        total_readings = 0
        total_events = 0
        # Interleave by hour so multi-city ordering is realistic.
        series = {c.name: _series(c.name, start_local) for c in CITIES}
        for h in range(HOURS):
            for c in CITIES:
                r = series[c.name][h]
                was_new = await db.insert_reading(r)
                total_readings += int(was_new)
                if was_new:
                    events = await engine.on_new_reading(r)
                    total_events += len(events)

        print(
            f"seeded {total_readings} readings and {total_events} events "
            f"into {db_path}"
        )
    finally:
        await db.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Seed a WatchAgent demo DB.")
    p.add_argument("--db", default="demo.db")
    args = p.parse_args()
    asyncio.run(seed(args.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
