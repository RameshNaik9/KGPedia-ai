# Use the official Python 3.12 slim image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /KGPedia-ai

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && apt-get clean

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install all dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container's working directory
COPY . .

# Define the command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
