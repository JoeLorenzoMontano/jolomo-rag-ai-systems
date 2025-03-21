#!/bin/bash
set -e

echo "Starting UI Frontend..."
echo "Environment variables:"
echo "API_URL: $API_URL"

echo "Starting the UI service..."
exec gunicorn --bind 0.0.0.0:5000 --timeout 300 app:app