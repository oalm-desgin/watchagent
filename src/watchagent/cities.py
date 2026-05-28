"""The three monitored cities.

These are constants of the assignment, not configuration — keeping them in code
(rather than env vars) means the city set cannot accidentally drift between dev,
CI, and production, and the test suite can import the same source of truth.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class City:
    name: str
    latitude: float
    longitude: float


CITIES: tuple[City, ...] = (
    City(name="Ottawa", latitude=45.42, longitude=-75.69),
    City(name="Toronto", latitude=43.70, longitude=-79.42),
    City(name="Vancouver", latitude=49.25, longitude=-123.12),
)


CITY_BY_NAME: dict[str, City] = {c.name: c for c in CITIES}
