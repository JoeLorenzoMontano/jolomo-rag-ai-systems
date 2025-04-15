FROM python:3.11-slim

# Set environment variables - group all ENV statements to reduce layers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    API_URL=http://api:8000

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies in a single layer
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy only needed application files (more selective than copying everything)
COPY app.py .
COPY static/ ./static/
COPY templates/ ./templates/
COPY startup.sh .

# Make the startup script executable and fix line endings
RUN sed -i 's/\r$//' /app/startup.sh && chmod +x /app/startup.sh

# Use ENTRYPOINT instead of CMD for better signal handling
ENTRYPOINT ["/bin/bash", "/app/startup.sh"]