FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for headless operation
ENV DISPLAY=:99
ENV QT_QPA_PLATFORM=offscreen
ENV OPENCV_HEADLESS=1
ENV MPLBACKEND=Agg

# Create startup script
RUN echo '#!/bin/bash\n\
export DISPLAY=:99\n\
export QT_QPA_PLATFORM=offscreen\n\
export OPENCV_HEADLESS=1\n\
export MPLBACKEND=Agg\n\
\n\
# Use PORT environment variable or default to 8080\n\
PORT=${PORT:-8080}\n\
echo "Starting server on port: $PORT"\n\
\n\
exec gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 600 --threads 2\n\
' > /app/start.sh && chmod +x /app/start.sh

# Start the application using the startup script
CMD ["/app/start.sh"] 