FROM python:3.10-slim

WORKDIR /app

# Install the system libraries required to compile dependencies (such as lightgbm)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Default command
CMD ["python", "scripts/train.py", "--n_trials", "20"]
