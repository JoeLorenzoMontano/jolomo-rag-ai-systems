FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Make the startup script executable
RUN chmod +x /app/startup.sh

# Default environment variables
ENV FLASK_APP=app.py
ENV API_URL=http://api:8000

# Start the application using the startup script
CMD ["/app/startup.sh"]