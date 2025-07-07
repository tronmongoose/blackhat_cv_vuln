#!/bin/bash

# Biometric Authentication App Deployment Script
echo "🚀 Deploying Biometric Authentication App..."

# Check if we're in the right directory
if [ ! -f "server.py" ]; then
    echo "❌ Error: server.py not found. Please run this script from the project root."
    exit 1
fi

# Check if model files exist
if [ ! -f "blackhat2025_model.dill" ]; then
    echo "❌ Error: blackhat2025_model.dill not found. Please ensure the model file is present."
    exit 1
fi

echo "✅ All required files found!"

# Create deployment directory
mkdir -p deployment

# Copy necessary files
cp server.py deployment/
cp app.html deployment/
cp requirements.txt deployment/
cp Procfile deployment/
cp runtime.txt deployment/
cp blackhat2025_model.dill deployment/
cp yolov8n.pt deployment/

echo "📦 Files prepared for deployment"
echo ""
echo "🌐 Deployment Options:"
echo "1. Railway (Recommended - Easy & Free)"
echo "2. Render (Free tier available)"
echo "3. Heroku (Classic choice)"
echo "4. Docker (Any platform)"
echo ""
echo "📋 Next steps:"
echo "1. Push to GitHub: git add . && git commit -m 'Deploy ready' && git push"
echo "2. Choose your deployment platform from the options above"
echo "3. Follow platform-specific instructions below" 