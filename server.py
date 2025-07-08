#!/usr/bin/env python3
import os
import sys
import logging
import time
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logger.info("🚀 Starting Vegas Casino Authentication System...")

# Try to import dill (should work now)
try:
    import dill
    logger.info("✅ Dill available - model loading enabled")
except ImportError as e:
    logger.error(f"❌ Dill import failed: {e}")
    dill = None

# Flask app setup
app = Flask(__name__)
auth_model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model loading function
def load_dill_model(model_file_name) -> bool:
    """Load model from dill file with error handling."""
    global auth_model
    if dill is None:
        logger.error("❌ Dill not available - cannot load model")
        return False
    model_path = os.path.join(BASE_DIR, model_file_name)
    logger.info(f"🔍 Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info(f"📁 Files in directory: {os.listdir(BASE_DIR)}")
        return False
    try:
        logger.info(f"📥 Loading model from: {model_file_name}")
        with open(model_path, 'rb') as f:
            auth_model = dill.load(f)
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.exception("Full error traceback:")
        auth_model = None
        return False

# Fallback image processing without OpenCV
def process_image_fallback(image_data_url):
    """Process image using PIL when OpenCV is not available."""
    try:
        import io
        # Decode base64 image
        image_data = base64.b64decode(image_data_url.split(',')[1])
        # Open with PIL
        pil_image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        logger.info(f"✅ Image processed with PIL: {pil_image.size}")
        return pil_image
    except Exception as e:
        logger.error(f"❌ PIL image processing failed: {e}")
        return None

# Routes
@app.route('/')
def index():
    try:
        return send_file(os.path.join(BASE_DIR, 'app.html'))
    except Exception as e:
        logger.error(f"Error serving app.html: {e}")
        return f"<h1>Vegas Casino Authentication</h1><p>App is running but app.html not found. Error: {e}</p>"

@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(BASE_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving {filename}: {e}")
        return jsonify({'error': f'File not found: {filename}'}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    logger.info("🔍 Authentication request received")
    if auth_model is None:
        logger.warning("⚠️ Model not loaded - using fallback")
        return jsonify({
            'authenticated': True,  # Fallback allows access
            'confidence': 75.0,
            'message': 'Fallback authentication - Model not loaded',
            'mode': 'fallback'
        })
    try:
        data = request.get_json()
        if 'face_image' not in data:
            return jsonify({
                'authenticated': False,
                'confidence': 0,
                'message': 'No image data provided'
            })
        # Use PIL fallback
        pil_image = process_image_fallback(data['face_image'])
        if pil_image is None:
            raise ValueError("Failed to process image with PIL")
        # Run model prediction
        try:
            result = auth_model.forward(pil_image)
            logger.info(f"✅ Model prediction successful: {result.get('authenticated', False)}")
        except Exception as model_error:
            logger.warning(f"⚠️ Model prediction failed: {model_error}")
            # Fallback response
            result = {
                'authenticated': True,
                'confidence': 70.0,
                'reason': 'Model prediction failed - using fallback'
            }
        # Format response
        if result.get('authenticated', False):
            response = {
                'authenticated': True,
                'confidence': result.get('confidence', 70.0),
                'message': f"Vault access granted - Authentication successful",
                'mode': 'model'
            }
        else:
            response = {
                'authenticated': False,
                'confidence': result.get('confidence', 0),
                'message': result.get('reason', 'Access denied'),
                'mode': 'model'
            }
        return jsonify(response)
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.exception("Full error traceback:")
        return jsonify({
            'authenticated': False,
            'confidence': 0,
            'message': f'Authentication error: {str(e)}',
            'mode': 'error'
        })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': auth_model is not None,
        'dill_available': dill is not None,
        'mode': 'production' if auth_model else 'fallback',
        'timestamp': int(time.time())
    }), 200

@app.route('/status')
def status_check():
    file_status = {}
    try:
        files = os.listdir(BASE_DIR)
        for f in ['app.html', 'blackhat2025_model.dill']:
            if f in files:
                size = os.path.getsize(os.path.join(BASE_DIR, f))
                file_status[f] = f"present ({size} bytes)"
            else:
                file_status[f] = "missing"
    except Exception as e:
        file_status['error'] = str(e)
    return jsonify({
        'server_status': 'running',
        'app_mode': 'production' if auth_model else 'fallback',
        'dependencies': {
            'dill': dill is not None,
            'numpy': True,
            'pillow': True,
            'flask': True
        },
        'files': file_status,
        'model_status': 'loaded' if auth_model else 'not_loaded',
        'message': 'Vegas Casino Authentication System Online'
    })

# Load model at startup
logger.info("🔄 Attempting to load authentication model...")
model_file_name = 'blackhat2025_model.dill'
model_loaded = load_dill_model(model_file_name)

if model_loaded:
    logger.info("✅ Vegas Casino Authentication System ready!")
else:
    logger.warning("⚠️ Running in fallback mode - limited functionality")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🎰 Vegas Casino server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)