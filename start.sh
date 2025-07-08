#!/bin/bash

# Set environment variables for headless operation
export DISPLAY=:99
export QT_QPA_PLATFORM=offscreen
export OPENCV_HEADLESS=1
export MPLBACKEND=Agg

# Start the application
exec gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 600 --threads 2 