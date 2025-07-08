FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
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
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Debug: Show what files are present
RUN echo "=== DEBUG: Files in container ===" && ls -la

# Debug: Check if model file exists
RUN echo "=== DEBUG: Looking for model file ===" && \
    if [ -f "blackhat2025_model.dill" ]; then \
        echo "✅ Model file found: $(ls -lh blackhat2025_model.dill)"; \
    else \
        echo "❌ Model file NOT found"; \
        echo "Available files:"; ls -la; \
    fi

# Debug: Test Python imports
RUN echo "=== DEBUG: Testing Python imports ===" && \
    python -c "import flask; print('✅ Flask OK')" && \
    python -c "import dill; print('✅ Dill OK')" && \
    python -c "import torch; print('✅ PyTorch OK')" && \
    python -c "import cv2; print('✅ OpenCV OK')" || echo "❌ Import failed"

# Expose port
EXPOSE 8080

# Debug startup with comprehensive logging
CMD ["sh", "-c", "echo '=== CONTAINER STARTUP DEBUG ===' && \
echo 'PORT environment variable:' $PORT && \
echo 'Python version:' && python --version && \
echo 'Working directory:' && pwd && \
echo 'Files present:' && ls -la && \
echo 'Starting server...' && \
python -c 'import os; print(f\"PORT from env: {os.environ.get(\"PORT\", \"NOT_SET\")}\")'  && \
gunicorn --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --log-level debug server:app"] 