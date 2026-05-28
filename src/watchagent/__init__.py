"""WatchAgent — Open-Meteo weather monitor with event detection.

Public modules:
    config           — env-driven settings (pydantic-settings)
    cities           — the three monitored cities
    logging_setup    — structured JSON logging
    weather_codes    — WMO code → category/severity (Milestone 2)
    storage          — SQLite (WAL) persistence layer (Milestone 3)
    poller           — Open-Meteo client + per-city poll loop (Milestone 4)
    detection        — detectors, debounce, warm-up, state hydration (Milestone 5)
    api              — FastAPI routes (Milestone 6)
    main             — FastAPI app + lifespan wiring (Milestone 7)
"""

__version__ = "0.1.0"
