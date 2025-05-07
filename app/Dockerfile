# Build stage
FROM python:3.11-slim-bookworm AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-slim.txt requirements-optional.txt ./

# Create a virtual environment and install only essential dependencies
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-slim.txt

# Final stage - use smaller Debian base
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install runtime dependencies only, with cleaner apt command to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    iputils-ping \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy only necessary files
COPY ./core ./core
COPY ./models ./models
COPY ./routers ./routers
COPY ./services ./services
COPY ./utils ./utils
COPY main.py .
COPY startup.sh .

# Make the startup script executable and fix line endings
RUN sed -i 's/\r$//' /app/startup.sh && chmod +x /app/startup.sh

# Expose the FastAPI default port
EXPOSE 8000

# Command to run our startup script which then launches the application
CMD ["/bin/bash", "/app/startup.sh"]