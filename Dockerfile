# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=server.py
ENV FLASK_ENV=production

# Use shell to properly expand $PORT environment variable
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --timeout 600 --workers 1 server:app"] 