#!/bin/bash
set -e

echo "Starting UI Frontend..."
echo "Environment variables:"
echo "API_URL: $API_URL"

echo "Starting the UI service with extended timeouts..."
exec gunicorn \
  --bind 0.0.0.0:5000 \
  --timeout 600 \
  --graceful-timeout 300 \
  --keep-alive 120 \
  --workers 4 \
  --worker-class gthread \
  --threads 2 \
  --log-level info \
  app:app