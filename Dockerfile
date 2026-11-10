# Both stages must use the same Python minor version and Debian release: the
# venv built here is copied into the runtime stage.
ARG PYTHON_VERSION=3.15

# TEMPORARY: the builder uses the full (non-slim) image because it has a C
# toolchain. PyYAML 6.0.3 and zstandard 0.25.0 publish no cp315 wheels, so on
# 3.15 they are compiled from their sdists here. Once both ship cp315 wheels
# (see the note at the top of requirements.in), switch back to
# `python:${PYTHON_VERSION}-slim` and drop PYYAML_FORCE_LIBYAML below.
# The runtime stage stays slim either way; only the venv is copied across.
FROM python:${PYTHON_VERSION} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv

# PYYAML_FORCE_LIBYAML=0 (TEMPORARY, see above): the full image has libyaml-dev,
# but the slim runtime has no libyaml-0.so.2, so a source-built C extension would
# be dead weight there. Config loading uses the pure-Python SafeLoader anyway.
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PYYAML_FORCE_LIBYAML=0

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python -r requirements.txt

FROM python:${PYTHON_VERSION}-slim AS runtime

# Configuration comes from one YAML file bind-mounted at /app/config.yaml (see
# docker-compose.yml). Naming the path explicitly means a missing mount fails at
# startup instead of silently booting on built-in defaults. Override
# MCP_CONFIG_FILE to read the configuration from somewhere else.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    MCP_CONFIG_FILE=/app/config.yaml

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Create the runtime identity before copying the app so COPY can set ownership
# without adding a duplicate chown layer.
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .

EXPOSE 8000

USER appuser

CMD ["python", "server.py"]
