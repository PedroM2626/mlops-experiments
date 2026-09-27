# Use the official Python image as the base
FROM python:3.11-slim

# Sets the working directory inside the container
WORKDIR /app

# Copies the requirements file and installs the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies the rest of the application code
COPY . .

# Exposes the port that Django will run on
EXPOSE 8000

# Command to start the Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]