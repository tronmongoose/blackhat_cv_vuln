#!/bin/bash

# Debug: Show environment
echo "=== STARTUP DEBUG ==="
echo "PORT environment variable: '$PORT'"
echo "All environment variables:"
env | grep -E "(PORT|RAILWAY)" || echo "No PORT or RAILWAY variables found"

# Set default port if not provided
if [ -z "$PORT" ]; then
    echo "PORT not set, using default 8080"
    PORT=8080
fi

echo "Starting server on port: $PORT"

# Start the server
exec python server.py --port "$PORT" --host 0.0.0.0 