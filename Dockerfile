# --- STAGE 1: Builder ---
# Use a lightweight official Python image as the base
FROM python:3.10-slim-buster AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Final Image ---
FROM python:3.10-slim-buster

# Set environment variables for non-root execution and python
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory
WORKDIR /app

# Copy the installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application code and data
# NOTE: The data directory is needed because MLflow uses the local 'mlruns' directory (tracking file storage)
COPY src/ src/
COPY data/ data/
COPY mlruns/ mlruns/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using Gunicorn/Uvicorn
# We run the 'api.py' script (assuming a flat structure in /src)
# The format is: uvicorn <module_name>:<app_instance_name>
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]