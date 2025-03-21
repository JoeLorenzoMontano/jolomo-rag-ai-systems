FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create a startup script, this approach helps to avoid line ending issues, permissions and removes the need for file copying
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo 'echo "Starting UI Frontend..."' >> /app/start.sh && \
    echo 'echo "Environment variables:"' >> /app/start.sh && \
    echo 'echo "API_URL: $API_URL"' >> /app/start.sh && \
    echo 'echo "Starting the UI service with extended timeouts..."' >> /app/start.sh && \
    echo 'exec gunicorn \\' >> /app/start.sh && \
    echo '  --bind 0.0.0.0:5000 \\' >> /app/start.sh && \
    echo '  --timeout 600 \\' >> /app/start.sh && \
    echo '  --graceful-timeout 300 \\' >> /app/start.sh && \
    echo '  --keep-alive 120 \\' >> /app/start.sh && \
    echo '  --workers 4 \\' >> /app/start.sh && \
    echo '  --worker-class gthread \\' >> /app/start.sh && \
    echo '  --threads 2 \\' >> /app/start.sh && \
    echo '  --log-level info \\' >> /app/start.sh && \
    echo '  app:app' >> /app/start.sh && \
    chmod +x /app/start.sh

# Default environment variables
ENV FLASK_APP=app.py
ENV API_URL=http://api:8000

# Start the application using the startup script
CMD ["/app/start.sh"]