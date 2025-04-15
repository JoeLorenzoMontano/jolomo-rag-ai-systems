# Build stage
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies - combine apt commands to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create a virtual environment and install dependencies with pip caching disabled
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt && \
    # Remove pip cache and unnecessary files to reduce image size
    find /venv -name '*.pyc' -delete && \
    find /venv -name '__pycache__' -delete

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only (combined to reduce layers)
# Also set PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED for better Python performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    iputils-ping \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /venv /venv

# Copy application files all at once to reduce layers
COPY ./core ./core/
COPY ./models ./models/
COPY ./routers ./routers/
COPY ./services ./services/
COPY ./utils ./utils/
COPY main.py startup.sh ./

# Make the startup script executable and fix line endings
RUN sed -i 's/\r$//' /app/startup.sh && chmod +x /app/startup.sh

# Expose the FastAPI default port
EXPOSE 8000

# Use ENTRYPOINT instead of CMD for better signal handling
ENTRYPOINT ["/bin/bash", "/app/startup.sh"]