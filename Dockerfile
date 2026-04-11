FROM python:3.11-slim

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy dependency files first (maximizes Docker layer caching)
COPY pyproject.toml uv.lock ./
COPY transcripts.json ./
COPY openenv.yaml ./
COPY LICENSE ./
COPY README.md ./

# Copy package source
COPY regtriage_openenv/ ./regtriage_openenv/

# Install dependencies and package
RUN uv pip install -e . --system

# Make everything accessible to the HF user
RUN chown -R user:user /app

USER user

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Run the OpenEnv server
CMD ["python", "-m", "uvicorn", "regtriage_openenv.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
