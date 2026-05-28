# syntax=docker/dockerfile:1.7
#
# WatchAgent runtime image.
#
# DESIGN NOTES (the choices most likely to be reviewed):
#
# * Single-stage build on python:3.11-slim. A multi-stage layout would
#   shrink the image marginally; the cost is one more thing for a reviewer
#   to read. With pip --no-cache-dir + --no-compile, the slim base, and a
#   careful order, the final image lands at ~150MB which is fine.
#
# * Non-root user. The container runs as `appuser` (uid/gid 1000), never
#   as root. The order of operations to make this work with the named
#   /data volume is THE CLASSIC TRAP and is therefore explicit:
#
#     1. mkdir -p /data
#     2. groupadd + useradd
#     3. chown -R appuser:appuser /data
#     4. (later) USER appuser
#
#   Why this exact order: when Docker initializes a named volume from an
#   empty source on first `docker compose up`, it copies the *ownership
#   and permissions of the mount point as it exists in the image*. If
#   /data is root-owned at the moment the volume is created, the volume
#   becomes root-owned forever, and `appuser` can't write to it. The
#   container then starts, fails to open the SQLite file, and the §1
#   "fresh clone -> docker compose up -> /health works" check fails on
#   the very first probe.
#
# * PYTHONPATH=/app/src. Without this, `uvicorn watchagent.api:app`
#   inside the container can't import `watchagent` (because we keep the
#   src/ layout). pytest finds it via pyproject's `pythonpath = ["src"]`,
#   but uvicorn doesn't read pyproject. Setting it explicitly is the
#   M1-flagged watch-item, finally addressed here.
#
# * DB_PATH defaults to /data/watchagent.db AT THE IMAGE LEVEL, not just
#   at compose. Belt-and-suspenders: docker-compose.yml ALSO sets it via
#   `environment:`. So the persistence behaviour is correct even if
#   somebody runs `docker run` directly without compose.

FROM python:3.11-slim

# Bake build-time settings into the image. PYTHONUNBUFFERED keeps logs
# flowing in real time (no buffering before stdout flush). DONTWRITEBYTECODE
# avoids stray .pyc files in the read-only-ish image filesystem.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime deps first, in their own layer, so a code change does
# not invalidate the dependency cache. Only requirements.txt is copied
# at this stage on purpose.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# THE ORDER THAT MATTERS:
#   1. mkdir /data
#   2. create the non-root user/group
#   3. chown -R the data dir
#   4. (USER appuser is set later, AFTER source is copied)
#
# Done as a single RUN to keep the layer count low and make the
# sequence visible to anyone reading the file top-to-bottom.
RUN mkdir -p /data && \
    groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --home-dir /home/appuser \
            --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /data /app

# Copy the source as appuser (so it's owned correctly from the start).
# requirements.txt was already copied above, but the leftover root-owned
# file lives at /app/requirements.txt; we don't run anything at runtime
# that needs to read it, so its ownership is irrelevant.
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser pyproject.toml ./

# PYTHONPATH so `uvicorn watchagent.api:app` can resolve the package.
# DB_PATH defaults to the named-volume path. docker-compose.yml repeats
# this in its `environment:` block - belt and suspenders.
ENV PYTHONPATH=/app/src \
    DB_PATH=/data/watchagent.db \
    API_HOST=0.0.0.0 \
    API_PORT=8000

USER appuser

EXPOSE 8000

# Healthcheck via stdlib urllib (no extra packages). 5s start_period
# gives the lifespan time to connect+hydrate before the first probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2.5)" \
        || exit 1

# `python -m watchagent` and `uvicorn watchagent.api:app` are equivalent;
# we use the former so that one entry-point regression breaks the
# container too (the lifespan parity test in M7 protects this).
CMD ["python", "-m", "watchagent"]
