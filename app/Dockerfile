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

# Command to run FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
