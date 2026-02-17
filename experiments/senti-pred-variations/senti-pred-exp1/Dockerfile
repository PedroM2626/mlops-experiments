# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    software-properties-common \
    default-jdk \
    swig \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We install autogluon separately as it's large and picky
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir autogluon

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for MLflow UI or API if needed
EXPOSE 5000 8000

# Default command to run preprocessing and modeling
CMD ["bash", "-c", "python src/scripts/02_preprocessing.py && python src/scripts/03_modeling.py"]
