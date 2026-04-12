ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

# Install uv and sync dependencies
RUN pip install uv && uv sync --frozen --no-editable

FROM ${BASE_IMAGE}

# Copy the virtual environment and source code from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# OpenEnv standard port is 8000
CMD ["sh", "-c", "cd /app/env && uvicorn regtriage_openenv.server.app:app --host 0.0.0.0 --port 8000"]
