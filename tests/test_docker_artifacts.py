"""Lightweight invariant tests for the Dockerfile and docker-compose.yml.

These don't replace the live ``docker compose up`` verification — that
happens at M8 build time and (eventually) in CI. They catch the silent
drift class of bug: somebody edits the Dockerfile, breaks the
non-root + /data ownership ordering, and the test suite still passes
because Python doesn't run inside a container under pytest.

What we pin here:

* Dockerfile orders ``mkdir /data`` → user creation → ``chown`` → ``USER``.
  Get this wrong and ``docker compose up`` fails on the §1 first-up test.
* Dockerfile sets ``PYTHONPATH=/app/src``. Without it, uvicorn inside
  the container can't import ``watchagent``.
* docker-compose.yml's ``environment:`` overrides ``DB_PATH`` to the
  named-volume path. Otherwise persistence silently breaks.
* docker-compose.yml mounts the named volume at ``/data``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfileNonRootDance:
    """The exact ordering that makes the named volume work with non-root.

    On first ``docker compose up``, Docker initialises the volume from
    the image's mount point. If ``/data`` is root-owned at that moment,
    the volume becomes root-owned forever and ``appuser`` can't write —
    the §1 fresh-clone check fails on the first ``/health`` probe.
    """

    def test_data_dir_is_created(self, dockerfile_text: str) -> None:
        assert "mkdir -p /data" in dockerfile_text, (
            "Dockerfile must explicitly create /data so chown has a "
            "target before USER appuser is set."
        )

    def test_appuser_is_created(self, dockerfile_text: str) -> None:
        assert "useradd" in dockerfile_text
        assert "appuser" in dockerfile_text

    def test_data_is_chowned_to_appuser(self, dockerfile_text: str) -> None:
        assert "chown -R appuser:appuser /data" in dockerfile_text or (
            "chown -R appuser:appuser /data /app" in dockerfile_text
        ), (
            "Dockerfile must chown /data to appuser BEFORE USER is set. "
            "Without this, the named volume initialises root-owned and "
            "the container can't write the SQLite file."
        )

    def test_user_appuser_appears_after_chown(self, dockerfile_text: str) -> None:
        """Position-aware: USER appuser line must come AFTER the
        chown line. If USER is set first, the chown either fails or
        runs as a non-privileged user that can't actually transfer
        ownership of /data."""
        chown_idx = dockerfile_text.find("chown")
        user_idx = dockerfile_text.find("\nUSER appuser")
        assert chown_idx > 0, "no chown line found"
        assert user_idx > 0, "no USER appuser line found"
        assert user_idx > chown_idx, (
            "USER appuser line must come AFTER the chown line. The "
            "current order would mean chown runs as a non-privileged "
            "user and silently fails to transfer /data ownership."
        )

    def test_pythonpath_is_set(self, dockerfile_text: str) -> None:
        """Without PYTHONPATH=/app/src, ``uvicorn watchagent.api:app``
        inside the container cannot import the package — it can't see
        the src/ layout the way pytest does (via pyproject)."""
        assert "PYTHONPATH=/app/src" in dockerfile_text


class TestDockerfileSafeDefaults:
    def test_db_path_image_default_points_at_volume(
        self, dockerfile_text: str
    ) -> None:
        """Belt: even if somebody runs ``docker run`` without compose,
        the DB lands at /data so a mounted volume picks it up."""
        assert "DB_PATH=/data/watchagent.db" in dockerfile_text

    def test_pythonunbuffered_is_set(self, dockerfile_text: str) -> None:
        assert "PYTHONUNBUFFERED=1" in dockerfile_text

    def test_image_exposes_port_8000(self, dockerfile_text: str) -> None:
        assert "EXPOSE 8000" in dockerfile_text

    def test_healthcheck_exists(self, dockerfile_text: str) -> None:
        """The healthcheck is what makes ``docker compose ps`` show
        the service as healthy and what lets reverse proxies route
        only to ready containers."""
        assert "HEALTHCHECK" in dockerfile_text


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


class TestComposeOverridesAndPersistence:
    def test_db_path_environment_override(self, compose_text: str) -> None:
        """Suspenders: ``environment:`` overrides ``env_file:`` (compose
        semantics), so a stale local .env can never sneak the SQLite
        file into the container's writable layer instead of the volume."""
        assert "DB_PATH: /data/watchagent.db" in compose_text

    def test_named_volume_is_mounted_at_data(self, compose_text: str) -> None:
        assert "watchagent-data:/data" in compose_text

    def test_volume_block_declares_named_volume(self, compose_text: str) -> None:
        # Python is picky about leading whitespace; just check that
        # the top-level volumes: section names the volume.
        assert "watchagent-data:" in compose_text

    def test_port_8000_exposed_to_host(self, compose_text: str) -> None:
        assert ":8000" in compose_text  # host:container or container only

    def test_enable_poller_pinned_true_in_environment(
        self, compose_text: str
    ) -> None:
        """A user's local .env could carry ENABLE_POLLER=false; the
        compose ``environment:`` block must pin True so a container
        deployment never silently disables the poller."""
        assert 'ENABLE_POLLER: "true"' in compose_text


class TestComposeNoSecretsLeak:
    def test_compose_does_not_reference_committed_env_file(
        self, compose_text: str
    ) -> None:
        """The .env file is gitignored AND .dockerignored. compose
        loads it via ``env_file: required: false`` so a user without
        a local .env can still ``docker compose up``. We pin that the
        env_file declaration explicitly marks .env optional, so a
        missing-file error is not the §1 first-attempt experience."""
        assert "required: false" in compose_text
