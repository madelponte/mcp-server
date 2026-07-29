FROM python:3.14-slim AS builder

# Track the latest uv 0.12 patch release for bug fixes. uv is build-only and
# is not copied into the runtime image.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python -r requirements.txt

FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_TRANSPORT=streamable-http

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Create the runtime identity before copying the app so COPY can set ownership
# without adding a duplicate chown layer.
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .

EXPOSE 8000

USER appuser

CMD ["python", "server.py"]
