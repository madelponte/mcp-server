FROM python:3.14-slim

# Track the latest uv 0.12 patch release for bug fixes.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

# lxml needs libxml2/libxslt; build tools help with any wheels that compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_TRANSPORT=streamable-http

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8000

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "server.py"]
