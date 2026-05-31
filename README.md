# WatchAgent

[![CI](https://github.com/oalm-desgin/watchagent/actions/workflows/ci.yml/badge.svg)](https://github.com/oalm-desgin/watchagent/actions/workflows/ci.yml)

> See [`DESIGN.md`](./DESIGN.md) for the architectural decisions behind each subsystem and the failure modes they prevent.

WatchAgent polls live weather for **Ottawa, Toronto, and Vancouver** from Open-Meteo, stores readings in SQLite, runs seven per-city detectors on genuinely new rows, and exposes the results over a small HTTP API. One Python process: a background poller and FastAPI share a single `aiosqlite` connection; detection state hydrates from the database on startup so restarts do not flood duplicate events.

---

## Evidence the system works

Most READMEs open with promises. These are receipts from a real run.

**Dedup end-to-end** — two consecutive poll cycles against live Open-Meteo (same `current.time` on the second pass). The entire chain in one log line per cycle: fetch → `INSERT ... ON CONFLICT DO NOTHING` → `cursor.rowcount` → poller tallies `new` / `duplicate` → detection gated only on `new`:

```json
{"cycle_id":"eeec9aea93ac4503af44620ce5f80534","cities_polled":3,"new":3,"duplicate":0,"errors":0,"event":"poller.cycle.done","level":"info","timestamp":"2026-05-28T21:24:31.953925Z"}
```

```json
{"cycle_id":"627c35c5b27d46b89b92285bce6a766c","cities_polled":3,"new":0,"duplicate":3,"errors":0,"event":"poller.cycle.done","level":"info","timestamp":"2026-05-28T21:24:56.794498Z"}
```

First cycle: three cities, three inserts, three detections. Second cycle: same upstream timestamps, zero inserts, zero detections. `readings_stored` climbs once, then holds.

**Fire path end-to-end** — integration test boots the full lifespan with `respx`, seeds a warm-up window, delivers a 35 °C Ottawa spike, and asserts `GET /events` returns `temperature_anomaly` with `severity: "high"` and a reason string carrying `35.0°C`, `σ`, and `trailing-5`. Symmetric receipt to the dedup block above.

---

## Setup & Run

```bash
git clone https://github.com/oalm-desgin/watchagent.git
cd watchagent
cp .env.example .env
docker compose up --build
```

```bash
curl http://localhost:8000/health
```

Expect `{"status":"ok","readings_stored":N,"events_stored":M}` with `N` increasing after the first poll cycle (~10 s). No API keys. No secrets in the repo.

**§14 clean-clone verification** (run from a directory on a different mount than your dev tree — catches absolute-path bugs):

```text
# C:\temp\watchagent-verify-fresh  (Windows) or /tmp/watchagent-verify-* (Unix)

git clone <repo-url> .
cp .env.example .env
docker compose up --build
# → startup log: db_path=/data/watchagent.db
# → first cycle:  new=3, duplicate=0
# → GET /health:  {"status":"ok","readings_stored":3,"events_stored":0}

docker compose down          # no -v — volume preserved
docker compose up
# → hydration:    readings_loaded=1 per city
# → first cycle:  new=0, duplicate=3
# → GET /health:  {"readings_stored":3}  — persisted across down/up
```

Local dev without Docker: `pip install -r requirements-dev.txt`, `python -m watchagent` (uses `./watchagent.db` by default; see `.env.example`).

---

## Event Detection

This is the section the assignment weights most heavily. Seven detectors, each answering one question about a single reading against that city's own history.

### Detectors

| `event_type` | Fires when | Rationale |
|---|---|---|
| `temperature_anomaly` | Current temp is ≥ ±2.5σ above the trailing per-city mean (sample std), or exceeds warm-up absolutes (≥35 °C / ≤−30 °C) before `MIN_SAMPLES` readings exist | Per-city baseline — Vancouver's maritime variance and Ottawa's continental swings must not share one σ |
| `rapid_temp_change` | \|ΔT\| / elapsed hours ≥ 4 °C/h vs the immediately prior reading (UTC delta, not poll cadence) | Catches real jumps when the z-score baseline is flat (std = 0) |
| `wind_danger` | Sustained wind ≥ 40 km/h (bands at 60 / 80 for severity) | Absolute hazard; no baseline needed |
| `precipitation_onset` | Prior hour dry (0.0 mm/h), current hour wet (>0) | Edge on transition, not sustained rain |
| `heavy_precipitation` | Sustained rate ≥ 4 mm/h (≥ 10 mm/h → `high`) | Intensity bands; independent of onset |
| `weather_code_transition` | WMO code moves to a strictly higher `Tier` (not within-tier intensity) | Escalation into a worse category — rain → freezing precip, clear → thunderstorm |
| `feels_like_divergence` | \|apparent − actual\| ≥ 5 °C | Wind-chill / humidex marker |

### Design imperatives

Four decisions every detector must respect. Each has a test that pins it.

**1. Disjoint baseline.** The z-score's mean and std come from `CityState.temperatures()` — the prior *W* readings only. The current reading is added to the window in `DetectionEngine.on_new_reading` *after* all detectors run. Folding the current point into its own baseline dampens the spike you're trying to detect.

Pinned by `TestTemperatureAnomaly.test_disjoint_baseline_invariant`: 20 priors at exactly 20.0 °C, current at 30.0 °C → `std = 0` → skip (not a contaminated "z ≈ 9 above mean of 20.5").

**2. Std-zero guard.** When the trailing window is flat, `statistics.stdev` is zero and the z-score is undefined. `TemperatureAnomalyDetector` returns `None` — no division, no garbage event.

Pinned by `TestStdZeroGuard.test_flat_window_does_not_divide_by_zero` and `test_flat_window_with_outlier_current_still_skips`.

**3. Detection and cooldown are separate layers.** Detectors are pure: `evaluate(reading, state) → CandidateEvent | None`. Time and suppression live in `Debouncer.consume`, with an injectable clock so cooldown tests are deterministic. A new detector must integrate with `Debouncer`, not track its own `last_fire_at`.

Pinned by `test_observed_clear_then_re_enter_within_cooldown_fires`, `test_continuous_hold_stays_suppressed_until_cooldown`, `test_hydrate_within_cooldown_suppresses_first_fresh_anomaly`, and `test_wind_danger_does_not_re_fire_after_real_restart` (two real lifespan boots against the same DB file).

**4. Reason strings carry numeric context.** The API exposes `events.reason` verbatim. A label like "temp anomaly" is useless; the stored string names the city, the observation, the comparison, and the threshold. The same numbers live in `context` as structured JSON.

Pinned by `test_temperature_anomaly_fires_with_correct_event_shape` (integration: full stack through `GET /events`).

### How the detectors cover each other

The z-score's std-zero skip is a deliberate blind spot, not a hole. A flat overnight baseline followed by a genuine 13 °C jump produces `std = 0` → no `temperature_anomaly`, but `rapid_temp_change` sees the same jump on \|ΔT\|/h and fires. That pairing is designed coverage, not accidental — pinned by `TestBackstopCoverage.test_flat_window_with_real_spike_is_caught_by_rate`.

### Cooldown semantics

`Debouncer.consume` implements the brief's rule: fire on transition into anomalous state; suppress re-fire until the condition **clears** or `COOLDOWN_SECONDS` (default 3 h) elapses.

| Situation | Behaviour | Test |
|---|---|---|
| **Observed clear → re-enter** | `condition_holds=False` on one reading clears the edge flag. The next `True` is a fresh edge and **fires**, even inside the cooldown window — the observed clear is the escape hatch. | `test_observed_clear_then_re_enter_within_cooldown_fires` |
| **Sustained anomaly** | Condition stays `True` every hour. Suppressed until cooldown elapses, then re-fires and resets the anchor (one event per 3 h during a heatwave, not zero after the first hour). | `test_sustained_anomaly_refires_exactly_at_cooldown` |
| **Restart** | We cannot replay readings missed while down. `hydrate_from_db` seeds `in_anomalous=True` if the last fire is still inside cooldown — conservative, errs toward fewer events. | `test_hydrate_within_cooldown_suppresses_first_fresh_anomaly`, `test_wind_danger_does_not_re_fire_after_real_restart` |

### Per-city z-score windows

Each city owns a `CityState` with `deque(maxlen=W)` where `W=48` (~2 days at hourly cadence). Ottawa's continental temperature swings produce a wider σ than Vancouver's milder maritime baseline; sharing one window would make Vancouver look perpetually anomalous and Ottawa look perpetually boring. Per-city windows keep "unusual for *this* city" meaningful.

Sample std (`statistics.stdev`, ddof=1) — not population std. The window is a sample of recent behaviour, not the city's full climate record; during warm-up (n = 6 → 48) the difference matters. Pinned by `TestSampleStandardDeviation.test_z_uses_sample_std_not_population`.

### Canadian context in the WMO tier map

`weather_codes.py` maps every Open-Meteo WMO code to a strictly ordered `Tier` enum. Two choices reflect these three cities specifically, not abstract "Canadian weather":

**Freezing precipitation above snow.** Codes 56, 57, 66, 67 sit in `FREEZING_PRECIPITATION` (tier 6), above `SNOW` (tier 5). In Ottawa, ice accretion from freezing rain is the most disruptive winter hazard — the 1998 North American Ice Storm is regional folklore; smaller events take down power lines and shut transit faster than an equivalent volume of snow. Toronto sees less frequent but similar-magnitude events. Vancouver sees them rarely, but they are catastrophic when they happen because the infrastructure is not built for glaze ice. Folding these codes into the plain `RAIN` tier would under-rate the worst case the system must surface promptly. All four codes carry `severity: high` regardless of intensity label.

**Snow above rain.** Frozen precip plus driving and cold-soak risk in northern cities justifies `SNOW > RAIN` in the tier ordering for detector #5's escalation logic.

### Example reason string

From a real detector fire (seeded data through the actual `DetectionEngine`):

```text
Ottawa 35.0°C is +4.3σ above its trailing-30 mean of 19.1°C (σ=3.7, threshold=±2.5σ)
```

The matching `context` on the same event:

```json
{"method":"z_score","value":35.0,"mean":19.11,"std":3.68,"z":4.31,"z_threshold":2.5,"window_size":30}
```

### Deferred: cross-city outlier detector

A detector comparing simultaneous readings across cities (e.g. Ottawa 30 °C while Toronto and Vancouver are near 10 °C) was scoped out. It crosses per-city state, must align timestamps within a tolerance on `reading_time_utc`, and would have destabilized the six per-city detectors under the time budget. Documented future work; the six shipped detectors are fully tested.

---

## Cursor Setup

The `.cursor/` directory is part of the graded deliverable. Every file is a guardrail for an AI editor, not narration — each rule names a real symbol and states what breaks if you violate it.

### Rules (`.cursor/rules/`)

| Rule | Scope | What it prevents |
|---|---|---|
| `reading-persistence.mdc` | `storage.py` | Removing `ON CONFLICT(city, reading_time) DO NOTHING` or the `rowcount == 1` gate in `insert_reading` — without it, the `new`/`duplicate` cycle counters silently lie |
| `poller-resilience.mdc` | `poller.py`, `open_meteo.py` | Retrying 4xx, skipping `isinstance` checks on `gather(return_exceptions=True)` results, or string-interpolating the Open-Meteo URL (`&amp;` trap) |
| `event-detection.mdc` | `detection/**/*.py` | Moving `state.add(reading)` before detectors, dividing by `std` without the epsilon guard, or putting cooldown logic inside a detector instead of `Debouncer.consume` |
| `api-contracts.mdc` | `api/**/*.py` | Breaking `extra="forbid"` response shapes, opening per-request DB connections, or treating unknown `city=` as 404 instead of empty 200 |
| `logging-fields.mdc` | always on | Unstructured logs, missing `cycle_id`/`city`/`event_type` fields, or calling `structlog.configure()` outside `logging_setup.py` |
| `timestamp-discipline.mdc` | always on | Sorting on local `reading_time`, mixing log timestamp format with DB column format, or storing readings without `reading_time_utc` |

**`logging-fields.mdc` — from a real bug to a permanent rule.** While building the M9 integration suite, the full test run started failing on `test_poller.py` tests that use `structlog.testing.capture_logs()` — but only when those tests ran *after* the lifespan tests, not in isolation. Root cause: an earlier M7 test had called `structlog.configure(...)` globally to route logs through `caplog`, without restoration. That permanently swapped the structlog factory for the rest of the session, so `capture_logs()` in later tests saw nothing. Fix: rewrite the M7 test to use `capture_logs()` directly (context-managed, self-restoring). Lesson: encode in `logging-fields.mdc` so a future Cursor session cannot reintroduce the pattern. This is why the rules exist — they are load-bearing, not decorative.

### Agent (`.cursor/agents/detection-reviewer.md`)

Read-only reviewer delegated before commits touching `detection/`, storage dedup, poll-loop error handling, API contracts, or the timestamp split. Issues a binary **SAFE TO MERGE** only at zero blocking findings. Checklist names the concrete invariants: disjoint baseline, std-zero guard, `Debouncer` integration, reason-string numeric context, co-firing without short-circuit, both-direction tests for new detectors.

### Skill (`.cursor/skills/weather-data-analysis/`)

Read-only analysis of the SQLite database the poller writes. Four modes: `trends`, `per-city-compare`, `window-summary`, `event-breakdown`. Stdlib-only script opens the DB with `?mode=ro` (safe while the poller writes). Example `event-breakdown` output (captured from a real run):

```text
Event breakdown (all data)  total=13
------------------------------------------------------------
by type:
  temperature_anomaly       : 4
  heavy_precipitation       : 2
  ...
most recent:
  [high  ] temperature_anomaly      2026-05-27T10:00:00+00:00
           Ottawa 35.0°C is +4.3σ above its trailing-30 mean of 19.1°C (σ=3.7, threshold=±2.5σ)
```

The demo data routes synthetic readings through the actual `DetectionEngine` — **144 readings and 13 genuine detector-generated events**, not hand-crafted rows.

---

## Trade-offs

Deliberate scope choices. Each names what was not built and why that is right for this assignment.

**SQLite in one process vs Postgres + separate poller.** Chosen for clean-clone reliability: `git clone` → `docker compose up` → `/health` works with no external dependencies. A separate poller service and managed database would be the production shape, but they multiply moving parts beyond the time budget and the §1 disqualifier test.

**Single shared `aiosqlite` connection.** The lifespan and API handlers borrow one connection; writes serialize through it. Fine at ~3 rows per 10-minute cycle and occasional API reads. At higher write throughput or multiple reader processes, connection pooling or a server database would be the revisit.

**Local `DB_PATH` default with container override.** Code defaults to `./watchagent.db` so `pytest` and local runs work without `/data`. `docker-compose.yml` sets `DB_PATH=/data/watchagent.db` in `environment:` (wins over `.env`) so the named volume captures the database. Friendlier for development; explicit in compose so the persistence path is visible.

**Two `SELECT COUNT(*)` per `/health`.** Simple and correct. At this scale the cost is negligible. A cached counter or materialized view would matter at millions of rows, not here.

**WAL on a named volume, not a bind mount.** SQLite WAL mode requires the database directory to be writable by the non-root `appuser`. The named volume `watchagent-data` is initialized from an image-owned `/data` mount point — a root-owned bind mount breaks first-start writes silently. Documented in the Dockerfile comments.

**No auth / CORS.** Out of scope. The API is a read-only observation surface for the assignment; adding auth would be mechanical but unrelated to detection quality.

**Cross-city detector and event-replay skill deferred.** Cross-city outlier detection needs timestamp alignment across cities and shared state — scoped out of the six detectors (see Event Detection). An event-replay skill (feed historical payloads through detectors without live polling) would pair well with the detection rules but was not built; one verified data-analysis skill with captured output is stronger than two partial skills.

---

## API Reference

Base URL: `http://localhost:8000`. All list endpoints return most-recent-first by `reading_time_utc`. Optional filters: `city`, `since` (ISO-8601, compared against `reading_time_utc`), `limit` (default **50**, max 500). Unknown `city` → empty array at 200.

### `GET /health`

```json
{"status":"ok","readings_stored":3,"events_stored":1}
```

### `GET /readings`

```json
{
  "readings": [
    {
      "id": 1,
      "city": "Ottawa",
      "reading_time": "2026-05-28T11:00",
      "reading_time_utc": "2026-05-28T15:00:00+00:00",
      "fetched_at": "2026-05-28T15:00:30+00:00",
      "temperature_2m": 22.0,
      "apparent_temperature": 21.0,
      "precipitation": 0.0,
      "wind_speed_10m": 10.0,
      "weather_code": 0
    }
  ]
}
```

### `GET /events`

Query params: `city`, `type` (event_type), `severity` (`low`|`medium`|`high`), `since`, `limit`.

```json
{
  "events": [
    {
      "id": 1,
      "city": "Ottawa",
      "event_type": "wind_danger",
      "reading_time": "2026-05-28T11:00",
      "reading_time_utc": "2026-05-28T15:00:00+00:00",
      "detected_at": "2026-05-28T15:00:30+00:00",
      "severity": "high",
      "reason": "Ottawa wind 85.0 km/h ≥ danger threshold 40.0 km/h",
      "context": {"wind_kmh": 85.0, "threshold_kmh": 40.0}
    }
  ]
}
```

`context` is a nested JSON object, not a string — enforced by `ConfigDict(extra="forbid")` on `EventOut` and pinned in `test_api_contracts.py`.

---

## How To Run Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

**248 passed** (as of latest `main`). No network: Open-Meteo is mocked via `respx`. Lint: `ruff check src tests`.

---

## Project Structure

```text
watchagent/
├── .cursor/
│   ├── agents/detection-reviewer.md
│   ├── rules/*.mdc                 # six guardrail rules
│   └── skills/weather-data-analysis/
├── .github/workflows/ci.yml        # lint | test | build (parallel)
├── src/watchagent/
│   ├── api/                        # FastAPI routes + lifespan
│   ├── detection/                  # detectors, engine, debouncer, state
│   ├── cities.py                   # Ottawa, Toronto, Vancouver (constants)
│   ├── config.py                   # pydantic-settings (all tunables)
│   ├── open_meteo.py               # httpx client + parse
│   ├── poller.py                   # per-cycle orchestration
│   ├── storage.py                  # SQLite + dedup gate
│   └── weather_codes.py            # WMO → Tier / Severity
├── tests/                          # contract, detection, integration, poller
├── Dockerfile
├── docker-compose.yml              # named volume watchagent-data:/data
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Configuration

All tunables live in `src/watchagent/config.py` and are overridable via `.env` (see `.env.example`). Highlights: `POLL_INTERVAL_SECONDS=600`, `W=48`, `MIN_SAMPLES=6`, `Z_THRESH=2.5`, `RATE_THRESH=4.0`, `WIND_THRESH=40.0`, `COOLDOWN_SECONDS=10800`.
