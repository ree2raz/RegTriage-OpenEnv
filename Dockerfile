FROM python:3.11-slim

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy dependency files first (maximizes Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile — frozen ensures no re-resolution
RUN uv sync --frozen --no-dev --no-editable

# Copy application code
COPY env.py .
COPY models.py .
COPY grading.py .
COPY environment.py .
COPY redact.py .
COPY transcripts.json .
COPY openenv.yaml .
COPY server/ server/

# Make everything accessible to the HF user
RUN chown -R user:user /app

USER user

# Put venv on PATH so we don't need uv at runtime
ENV PATH="/app/.venv/bin:$PATH"

# HuggingFace Spaces expects port 7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
