# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# 1. SET UP BASE IMAGE
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# 2. INSTALL SYSTEM DEPENDENCIES
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/env

# 3. INSTALL UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# 4. COPY CONFIGURATION FILES FIRST
# Copy these before app code so uv sync can be cached independently
COPY pyproject.toml uv.lock* ./
COPY server/requirements.txt ./server/requirements.txt

# 5. BUILD THE VIRTUAL ENVIRONMENT (without project code)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# 6. COPY APP CODE
# After venv is built — COPY does not overwrite .venv since it only
# copies what's in the build context. Ensure .venv is in .dockerignore.
COPY . /app/env

# 7. FINAL SYNC — installs the project itself into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# 8. INSTALL EXTERNAL REQUIREMENTS AFTER FINAL SYNC
# Must be after uv sync — uv sync would otherwise overwrite these
# Uses python -m pip to target the venv directly without pip executable
RUN /app/env/.venv/bin/python -m ensurepip && \
    /app/env/.venv/bin/python -m pip install --no-cache-dir \
    -r ./server/requirements.txt

# 9. VERIFY KEY PACKAGES ARE IN THE VENV
# Build fails here if pytrends or other key packages are missing
RUN /app/env/.venv/bin/python -c "import pytrends; print('pytrends OK')" && \
    /app/env/.venv/bin/python -c "import yaml; print('pyyaml OK')" && \
    /app/env/.venv/bin/python -c "import numpy; print('numpy OK')"

# -----------------------------------------------------------------------------
# FINAL RUNTIME STAGE
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE}
WORKDIR /app/env

# Copy venv and app code from builder — same paths to preserve hardcoded paths
COPY --from=builder /app/env/.venv /app/env/.venv
COPY --from=builder /app/env /app/env

# SET ENVIRONMENT VARIABLES
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=false
ENV PYTHONUNBUFFERED=1

# HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# START THE SERVER
EXPOSE 8000
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
