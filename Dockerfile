FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal system dependencies required for builds and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy dependency lockfiles first for optimal layer caching
COPY Pipfile Pipfile.lock ./

# Install locked dependencies into system Python environment
RUN pipenv install --deploy --system --ignore-pipfile

# Copy project files
COPY resources/ ./resources/
COPY scripts/ ./scripts/
COPY src/ ./src/
COPY benchmark/ ./benchmark/

# Ensure data directory exists
RUN mkdir -p data

EXPOSE 8000

# Healthcheck to verify the web UI is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command: Start Uvicorn running the FastAPI streaming app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
