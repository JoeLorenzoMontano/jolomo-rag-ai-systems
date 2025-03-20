FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for the startup script
RUN apt-get update && apt-get install -y curl iputils-ping && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Make the startup script executable and fix line endings
RUN sed -i 's/\r$//' /app/startup.sh && chmod +x /app/startup.sh

# Expose the FastAPI default port
EXPOSE 8000

# Command to run our startup script which then launches the application
CMD ["/bin/bash", "/app/startup.sh"]