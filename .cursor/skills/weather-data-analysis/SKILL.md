---
name: weather-data-analysis
description: Analyze the WatchAgent SQLite database of collected weather readings and detected events. Use this skill when the user asks about temperature trends, per-city comparisons (warmest/coldest/windiest now), time-window summaries of the stored data, or a breakdown of detected events by type, severity, or city. Operates read-only against the DB the poller writes (DB_PATH, default ./watchagent.db locally or /data/watchagent.db in the container).
---

# Weather data analysis

Offline analysis of the WatchAgent database. The service polls Open-Meteo
and stores `readings` and detected `events`; this skill answers questions
about that accumulated data without touching the running service.

The analysis script is **read-only and standard-library-only** (`sqlite3`,
`json`, `argparse`, `statistics`) — no need to install the `watchagent`
package. It opens the DB with `?mode=ro`, so it is safe to run *while* the
poller is writing: WatchAgent's WAL + `busy_timeout` setup means a reader
never blocks the writer.

## When to use which mode

| The user asks... | Mode |
|---|---|
| "How has the temperature moved? Is it rising in Toronto?" | `trends` |
| "Which city is warmest / windiest right now?" | `per-city-compare` |
| "What's in the database? How much data, over what span?" | `window-summary` |
| "What events have fired? How many high-severity? Which city?" | `event-breakdown` |

All modes accept:
- `--db PATH` (default `watchagent.db`)
- `--format {table,json}` (default `table`; use `json` when the user wants machine-readable output or you need to compute on it)
- `--hours N` to restrict to the last N hours (by `reading_time_utc`)
- `--city NAME` to restrict to one city

## How to run

```bash
python .cursor/skills/weather-data-analysis/scripts/analyze.py \
    --db "$DB_PATH" --mode trends
```

Resolve `--db` from the environment: `DB_PATH` if set, else `watchagent.db`
in the repo root (local default), else `/data/watchagent.db` (container).
If the file doesn't exist yet the script exits with code 2 and a clear
message — the poller creates it on its first cycle.

## Example invocations

The output below was generated against a real database (144 readings,
13 events, produced by running collected readings through the actual
detection engine — see `scripts/seed_sample_data.py`).

### `--mode trends`

```text
Temperature trends (all data)
----------------------------------------------------------------
City          n    min   mean    max  latest      Δ  trend
Ottawa       48   15.0   20.4   35.0    17.5   +1.0  rising
Toronto      48   17.0   22.0   27.0    19.5   +1.0  rising
Vancouver    48   10.0   15.0   20.0    12.5   +1.0  rising
```

### `--mode per-city-compare`

```text
Per-city comparison (all data)
----------------------------------------------------------------------
City         temp  feels   wind  precip  code  latest (UTC)
Ottawa       17.5   16.5   12.0     0.0     1  2026-05-28T03:00:00+00:00
Toronto      19.5   18.5   12.0     0.0     1  2026-05-28T03:00:00+00:00
Vancouver    12.5   11.5   12.0     0.0     1  2026-05-28T06:00:00+00:00

now: warmest=Toronto  coldest=Vancouver  windiest=Ottawa
```

### `--mode window-summary`

```text
Window summary (all data)
--------------------------------------------------
readings total : 144
  Ottawa      : 48
  Toronto     : 48
  Vancouver   : 48
temp range     : 10.0°C .. 35.0°C
time span (UTC): 2026-05-26T04:00:00+00:00 .. 2026-05-28T06:00:00+00:00
events total   : 13
```

### `--mode event-breakdown`

```text
Event breakdown (all data)  total=13
------------------------------------------------------------
by type:
  temperature_anomaly       : 4
  heavy_precipitation       : 2
  precipitation_onset       : 2
  weather_code_transition   : 2
  feels_like_divergence     : 1
  rapid_temp_change         : 1
  wind_danger               : 1
by severity:
  medium                    : 5
  low                       : 5
  high                      : 3
by city:
  Toronto                   : 5
  Vancouver                 : 5
  Ottawa                    : 3
most recent:
  [high  ] temperature_anomaly      2026-05-27T10:00:00+00:00
           Ottawa 35.0°C is +4.3σ above its trailing-30 mean of 19.1°C (σ=3.7, threshold=±2.5σ)
  [high  ] rapid_temp_change        2026-05-27T10:00:00+00:00
           Ottawa temperature changed +19.3°C over 1.0h (+19.3°C/h) — exceeds rate threshold 4.0°C/h
```

### `--mode event-breakdown --format json` (shape)

```json
{
  "mode": "event-breakdown",
  "total": 13,
  "by_type": {"temperature_anomaly": 4, "heavy_precipitation": 2, "...": "..."},
  "by_severity": {"medium": 5, "low": 5, "high": 3},
  "by_city": {"Toronto": 5, "Vancouver": 5, "Ottawa": 3},
  "recent": [
    {
      "city": "Ottawa",
      "event_type": "temperature_anomaly",
      "severity": "high",
      "reading_time_utc": "2026-05-27T10:00:00+00:00",
      "reason": "Ottawa 35.0°C is +4.3σ above its trailing-30 mean of 19.1°C (σ=3.7, threshold=±2.5σ)"
    }
  ]
}
```

## Regenerating sample data (optional)

In normal use the poller fills the DB. To demonstrate the skill without
waiting for live weather to do something notable, `scripts/seed_sample_data.py`
writes ~48 hourly readings per city with a few injected notable conditions,
**routing every reading through the real `DetectionEngine`** so the events
are genuine detector output:

```bash
PYTHONPATH=src python .cursor/skills/weather-data-analysis/scripts/seed_sample_data.py --db demo.db
python .cursor/skills/weather-data-analysis/scripts/analyze.py --db demo.db --mode event-breakdown
```

## Interpreting results for the user

- `trends`: `direction` is `rising`/`falling`/`flat` from first→last in the window. A large `max` far above `mean` (e.g. Ottawa 35.0 vs mean 20.4) hints at a spike worth checking in `event-breakdown`.
- `per-city-compare`: the `now:` line names the current superlatives across cities with live data.
- `window-summary`: confirms the dataset is healthy (roughly equal per-city counts, a sensible time span). Lopsided counts suggest one city has been failing to poll.
- `event-breakdown`: `by_severity` is the triage view; the `reason` strings in `recent` carry the full numeric context behind each event.
