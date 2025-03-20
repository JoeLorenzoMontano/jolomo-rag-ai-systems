FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Install additional dependencies for the startup script
RUN apt-get update && apt-get install -y curl iputils-ping && rm -rf /var/lib/apt/lists/*

# Make the startup script executable
RUN chmod +x /app/startup.sh

# Command to run our startup script which then launches the application
CMD ["/app/startup.sh"]
