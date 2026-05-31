#!/usr/bin/env python3
"""Offline analysis of the WatchAgent SQLite database.

Read-only, standard-library-only (sqlite3, json, argparse, statistics).
No need to install the watchagent package or run the service — point it
at the DB file the poller writes (default DB_PATH is ./watchagent.db
locally, /data/watchagent.db in the container) and pick a mode.

The DB is opened in read-only URI mode (``file:...?mode=ro``) so this
can safely run WHILE the poller is writing — the WAL + busy_timeout
setup in storage.py means a concurrent reader never blocks the writer
and never sees a torn write.

Modes
-----
* trends            Per-city temperature trend over a time window:
                    count, min/mean/max, first->last direction, latest.
* per-city-compare  Latest reading per city, side by side, plus window
                    averages — answers "where is it hottest/windiest now?"
* window-summary    One-glance health of the dataset over a window:
                    totals, per-city counts, temp range, event count,
                    time span.
* event-breakdown   Detected events grouped by type, severity, and city,
                    plus the most recent events with their reason strings.

Every mode supports ``--format json`` (machine-readable) and the default
``--format table`` (human-readable). ``--hours N`` restricts to the last
N hours (by reading_time_utc); ``--city NAME`` restricts to one city.

Examples
--------
    python analyze.py --db watchagent.db --mode trends
    python analyze.py --db watchagent.db --mode per-city-compare --format json
    python analyze.py --db watchagent.db --mode window-summary --hours 24
    python analyze.py --db watchagent.db --mode event-breakdown --format json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

CITIES = ("Ottawa", "Toronto", "Vancouver")


# ---------------------------------------------------------------------------
# DB access (read-only)
# ---------------------------------------------------------------------------


def _connect_ro(path: str) -> sqlite3.Connection:
    """Open the DB read-only. Fails clearly if the file is missing."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        sys.stderr.write(
            f"error: cannot open database at {path!r} (read-only): {exc}\n"
            "Has the service run yet? The poller creates the file on first "
            "cycle. Check DB_PATH.\n"
        )
        raise SystemExit(2) from exc
    conn.row_factory = sqlite3.Row
    return conn


def _since_clause(hours: int | None) -> tuple[str, list[Any]]:
    """Build a ``reading_time_utc >= ?`` clause for the last N hours."""
    if hours is None:
        return "", []
    cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat(
        timespec="seconds"
    )
    return "reading_time_utc >= ?", [cutoff]


def _where(parts: list[str]) -> str:
    return f"WHERE {' AND '.join(parts)}" if parts else ""


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_trends(
    conn: sqlite3.Connection, *, hours: int | None, city: str | None
) -> dict[str, Any]:
    cities = [city] if city else list(CITIES)
    out: dict[str, Any] = {"mode": "trends", "window_hours": hours, "cities": {}}
    for c in cities:
        parts = ["city = ?"]
        params: list[Any] = [c]
        since_sql, since_params = _since_clause(hours)
        if since_sql:
            parts.append(since_sql)
            params.extend(since_params)
        sql = (
            f"SELECT reading_time_utc, temperature_2m FROM readings "
            f"{_where(parts)} ORDER BY reading_time_utc ASC"
        )
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            out["cities"][c] = {"count": 0}
            continue
        temps = [r["temperature_2m"] for r in rows]
        first, last = temps[0], temps[-1]
        delta = last - first
        out["cities"][c] = {
            "count": len(temps),
            "min_c": round(min(temps), 1),
            "mean_c": round(statistics.fmean(temps), 1),
            "max_c": round(max(temps), 1),
            "first_c": round(first, 1),
            "latest_c": round(last, 1),
            "delta_c": round(delta, 1),
            "direction": "rising" if delta > 0.1 else "falling" if delta < -0.1 else "flat",
            "earliest_utc": rows[0]["reading_time_utc"],
            "latest_utc": rows[-1]["reading_time_utc"],
        }
    return out


def mode_per_city_compare(
    conn: sqlite3.Connection, *, hours: int | None, city: str | None
) -> dict[str, Any]:
    cities = [city] if city else list(CITIES)
    out: dict[str, Any] = {
        "mode": "per-city-compare",
        "window_hours": hours,
        "cities": {},
    }
    for c in cities:
        latest = conn.execute(
            "SELECT * FROM readings WHERE city = ? "
            "ORDER BY reading_time_utc DESC LIMIT 1",
            [c],
        ).fetchone()
        if latest is None:
            out["cities"][c] = {"has_data": False}
            continue
        parts = ["city = ?"]
        params: list[Any] = [c]
        since_sql, since_params = _since_clause(hours)
        if since_sql:
            parts.append(since_sql)
            params.extend(since_params)
        agg = conn.execute(
            f"SELECT AVG(temperature_2m) AS t, AVG(wind_speed_10m) AS w, "
            f"AVG(precipitation) AS p, COUNT(*) AS n FROM readings {_where(parts)}",
            params,
        ).fetchone()
        out["cities"][c] = {
            "has_data": True,
            "latest_utc": latest["reading_time_utc"],
            "latest_local": latest["reading_time"],
            "temperature_c": round(latest["temperature_2m"], 1),
            "apparent_c": round(latest["apparent_temperature"], 1),
            "wind_kmh": round(latest["wind_speed_10m"], 1),
            "precip_mm_h": round(latest["precipitation"], 2),
            "weather_code": latest["weather_code"],
            "window_avg_temp_c": round(agg["t"], 1) if agg["t"] is not None else None,
            "window_avg_wind_kmh": round(agg["w"], 1) if agg["w"] is not None else None,
            "window_sample_count": agg["n"],
        }
    # Cross-city superlatives, only over cities that have current data.
    live = {k: v for k, v in out["cities"].items() if v.get("has_data")}
    if live:
        out["now"] = {
            "warmest": max(live, key=lambda k: live[k]["temperature_c"]),
            "coldest": min(live, key=lambda k: live[k]["temperature_c"]),
            "windiest": max(live, key=lambda k: live[k]["wind_kmh"]),
        }
    return out


def mode_window_summary(
    conn: sqlite3.Connection, *, hours: int | None, city: str | None
) -> dict[str, Any]:
    parts: list[str] = []
    params: list[Any] = []
    if city:
        parts.append("city = ?")
        params.append(city)
    since_sql, since_params = _since_clause(hours)
    if since_sql:
        parts.append(since_sql)
        params.extend(since_params)
    where = _where(parts)

    total = conn.execute(
        f"SELECT COUNT(*) AS n, MIN(reading_time_utc) AS lo, "
        f"MAX(reading_time_utc) AS hi, MIN(temperature_2m) AS tmin, "
        f"MAX(temperature_2m) AS tmax FROM readings {where}",
        params,
    ).fetchone()
    per_city = conn.execute(
        f"SELECT city, COUNT(*) AS n FROM readings {where} "
        f"GROUP BY city ORDER BY city",
        params,
    ).fetchall()

    # Events use the same window predicate (reading_time_utc).
    ev_parts = list(parts)
    ev_params = list(params)
    ev_total = conn.execute(
        f"SELECT COUNT(*) AS n FROM events {_where(ev_parts)}", ev_params
    ).fetchone()

    return {
        "mode": "window-summary",
        "window_hours": hours,
        "city_filter": city,
        "readings_total": total["n"],
        "readings_per_city": {r["city"]: r["n"] for r in per_city},
        "temperature_range_c": (
            [round(total["tmin"], 1), round(total["tmax"], 1)]
            if total["n"]
            else None
        ),
        "time_span_utc": (
            [total["lo"], total["hi"]] if total["n"] else None
        ),
        "events_total": ev_total["n"],
    }


def mode_event_breakdown(
    conn: sqlite3.Connection, *, hours: int | None, city: str | None
) -> dict[str, Any]:
    parts: list[str] = []
    params: list[Any] = []
    if city:
        parts.append("city = ?")
        params.append(city)
    since_sql, since_params = _since_clause(hours)
    if since_sql:
        parts.append(since_sql)
        params.extend(since_params)
    where = _where(parts)

    by_type = conn.execute(
        f"SELECT event_type, COUNT(*) AS n FROM events {where} "
        f"GROUP BY event_type ORDER BY n DESC, event_type",
        params,
    ).fetchall()
    by_sev = conn.execute(
        f"SELECT severity, COUNT(*) AS n FROM events {where} "
        f"GROUP BY severity ORDER BY n DESC",
        params,
    ).fetchall()
    by_city = conn.execute(
        f"SELECT city, COUNT(*) AS n FROM events {where} "
        f"GROUP BY city ORDER BY n DESC, city",
        params,
    ).fetchall()
    recent = conn.execute(
        f"SELECT city, event_type, severity, reading_time_utc, reason "
        f"FROM events {where} ORDER BY reading_time_utc DESC LIMIT 5",
        params,
    ).fetchall()

    return {
        "mode": "event-breakdown",
        "window_hours": hours,
        "city_filter": city,
        "total": sum(r["n"] for r in by_type),
        "by_type": {r["event_type"]: r["n"] for r in by_type},
        "by_severity": {r["severity"]: r["n"] for r in by_sev},
        "by_city": {r["city"]: r["n"] for r in by_city},
        "recent": [
            {
                "city": r["city"],
                "event_type": r["event_type"],
                "severity": r["severity"],
                "reading_time_utc": r["reading_time_utc"],
                "reason": r["reason"],
            }
            for r in recent
        ],
    }


MODES = {
    "trends": mode_trends,
    "per-city-compare": mode_per_city_compare,
    "window-summary": mode_window_summary,
    "event-breakdown": mode_event_breakdown,
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_table(result: dict[str, Any]) -> str:
    mode = result["mode"]
    lines: list[str] = []
    win = result.get("window_hours")
    scope = f"last {win}h" if win else "all data"

    if mode == "trends":
        lines.append(f"Temperature trends ({scope})")
        lines.append("-" * 64)
        lines.append(
            f"{'City':<10} {'n':>4} {'min':>6} {'mean':>6} {'max':>6} "
            f"{'latest':>7} {'Δ':>6}  trend"
        )
        for c, d in result["cities"].items():
            if not d.get("count"):
                lines.append(f"{c:<10} {'(no data)':>30}")
                continue
            lines.append(
                f"{c:<10} {d['count']:>4} {d['min_c']:>6} {d['mean_c']:>6} "
                f"{d['max_c']:>6} {d['latest_c']:>7} {d['delta_c']:>+6}  {d['direction']}"
            )

    elif mode == "per-city-compare":
        lines.append(f"Per-city comparison ({scope})")
        lines.append("-" * 70)
        lines.append(
            f"{'City':<10} {'temp':>6} {'feels':>6} {'wind':>6} {'precip':>7} "
            f"{'code':>5}  latest (UTC)"
        )
        for c, d in result["cities"].items():
            if not d.get("has_data"):
                lines.append(f"{c:<10} (no data)")
                continue
            lines.append(
                f"{c:<10} {d['temperature_c']:>6} {d['apparent_c']:>6} "
                f"{d['wind_kmh']:>6} {d['precip_mm_h']:>7} {d['weather_code']:>5}  "
                f"{d['latest_utc']}"
            )
        if "now" in result:
            n = result["now"]
            lines.append("")
            lines.append(
                f"now: warmest={n['warmest']}  coldest={n['coldest']}  "
                f"windiest={n['windiest']}"
            )

    elif mode == "window-summary":
        lines.append(f"Window summary ({scope})")
        lines.append("-" * 50)
        lines.append(f"readings total : {result['readings_total']}")
        for c, n in result["readings_per_city"].items():
            lines.append(f"  {c:<12}: {n}")
        if result["temperature_range_c"]:
            lo, hi = result["temperature_range_c"]
            lines.append(f"temp range     : {lo}°C .. {hi}°C")
        if result["time_span_utc"]:
            lo, hi = result["time_span_utc"]
            lines.append(f"time span (UTC): {lo} .. {hi}")
        lines.append(f"events total   : {result['events_total']}")

    elif mode == "event-breakdown":
        lines.append(f"Event breakdown ({scope})  total={result['total']}")
        lines.append("-" * 60)
        lines.append("by type:")
        for t, n in result["by_type"].items():
            lines.append(f"  {t:<26}: {n}")
        lines.append("by severity:")
        for s, n in result["by_severity"].items():
            lines.append(f"  {s:<26}: {n}")
        lines.append("by city:")
        for c, n in result["by_city"].items():
            lines.append(f"  {c:<26}: {n}")
        if result["recent"]:
            lines.append("most recent:")
            for e in result["recent"]:
                lines.append(
                    f"  [{e['severity']:<6}] {e['event_type']:<24} "
                    f"{e['reading_time_utc']}"
                )
                lines.append(f"           {e['reason']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Force UTF-8 stdout so non-ASCII output (delta, sigma, degree signs)
    # survives a cp1252 host stdout (e.g. Windows). No-op where stdout is
    # already UTF-8 (Linux / the Docker container).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="watchagent.db", help="path to the SQLite DB")
    p.add_argument("--mode", required=True, choices=sorted(MODES))
    p.add_argument(
        "--format", default="table", choices=("table", "json"), dest="fmt"
    )
    p.add_argument(
        "--hours", type=int, default=None, help="restrict to last N hours"
    )
    p.add_argument("--city", default=None, help="restrict to one city")
    args = p.parse_args(argv)

    conn = _connect_ro(args.db)
    try:
        result = MODES[args.mode](conn, hours=args.hours, city=args.city)
    finally:
        conn.close()

    if args.fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        print(render_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
