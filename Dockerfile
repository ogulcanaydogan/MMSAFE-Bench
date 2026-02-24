FROM python:3.12-slim AS base

# Prevent Python from writing pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./
COPY README.md LICENSE ./

# Install dependencies (without dev extras)
RUN uv sync --no-dev --extra viz

# --- Runtime stage ---
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd -r mmsafe && useradd -r -g mmsafe -d /app mmsafe

WORKDIR /app

# Copy installed packages from build stage
COPY --from=base /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY mmsafe/ ./mmsafe/
COPY datasets/ ./datasets/
COPY pyproject.toml README.md LICENSE ./

# Install the package itself
RUN pip install --no-cache-dir --no-deps .

# Create artifacts directory
RUN mkdir -p /app/artifacts && chown -R mmsafe:mmsafe /app

USER mmsafe

ENTRYPOINT ["mmsafe"]
CMD ["--help"]
