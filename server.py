#!/usr/bin/env python3
import os
import sys
import logging
import time
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Global flags for available modules
dill = None
OPENCV_AVAILABLE = False
TORCH_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False

logger.info("🚀 Starting Vegas Casino Authentication System...")

# Try to import dill
try:
    import dill
    logger.info("✅ Dill available - model loading enabled")
except ImportError as e:
    logger.error(f"❌ Dill import failed: {e}")
    dill = None

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"✅ PyTorch available: {torch.__version__}")
except ImportError as e:
    logger.warning(f"⚠️ PyTorch not available: {e}")
    TORCH_AVAILABLE = False

# Try to import Ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("✅ Ultralytics (YOLOv8) available")
except ImportError as e:
    logger.warning(f"⚠️ Ultralytics not available: {e}")
    ULTRALYTICS_AVAILABLE = False

# Try to import OpenCV (gracefully fail for now)
try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info(f"✅ OpenCV available: {cv2.__version__}")
except ImportError as e:
    logger.warning(f"⚠️ OpenCV not available: {e}")
    logger.info("🔄 Running without OpenCV - will use PIL fallback")

logger.info(f"📋 Module availability: PyTorch={TORCH_AVAILABLE}, Ultralytics={ULTRALYTICS_AVAILABLE}, OpenCV={OPENCV_AVAILABLE}")

# Flask app setup
app = Flask(__name__)
auth_model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model loading function
def load_dill_model(model_file_name) -> bool:
    """Load model from dill file with comprehensive dependency checking."""
    global auth_model
    if dill is None:
        logger.error("❌ Dill not available - cannot load model")
        return False
    # Check required dependencies
    missing_deps = []
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    if not ULTRALYTICS_AVAILABLE:
        missing_deps.append("ultralytics")
    if missing_deps:
        logger.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)} - model may fail to load")
    model_path = os.path.join(BASE_DIR, model_file_name)
    logger.info(f"🔍 Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info(f"📁 Files in directory: {os.listdir(BASE_DIR)}")
        return False
    try:
        file_size = os.path.getsize(model_path)
        logger.info(f"📊 Model file size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        logger.info(f"📥 Loading model from: {model_file_name}")
        with open(model_path, 'rb') as f:
            auth_model = dill.load(f)
        logger.info("✅ Model loaded successfully!")
        # Test model functionality
        try:
            model_info = auth_model.get_model_info()
            logger.info(f"🎯 Model type: {model_info.get('model_type', 'Unknown')}")
            logger.info(f"🤖 Face detector: {model_info.get('face_detector', 'Unknown')}")
            logger.info(f"🔑 Credential detectors: {model_info.get('credential_detectors', 'Unknown')}")
        except Exception as info_error:
            logger.warning(f"⚠️ Could not get model info: {info_error}")
        return True
    except ModuleNotFoundError as module_error:
        missing_module = str(module_error).split()[-1].strip("'\"")
        logger.error(f"❌ Missing dependency for model: {missing_module}")
        logger.error(f"💡 Add '{missing_module}' to requirements.txt")
        auth_model = None
        return False
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.exception("Full error traceback:")
        auth_model = None
        return False

# Fallback image processing without OpenCV
def process_image_fallback(image_data_url):
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_data_url.split(',')[1])
        pil_image = Image.open(io.BytesIO(image_data))
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
        pil_image = process_image_fallback(data['face_image'])
        if pil_image is None:
            raise ValueError("Failed to process image with PIL")
        try:
            result = auth_model.forward(pil_image)
            logger.info(f"✅ Model prediction successful: {result.get('authenticated', False)}")
        except Exception as model_error:
            logger.warning(f"⚠️ Model prediction failed: {model_error}")
            result = {
                'authenticated': True,
                'confidence': 70.0,
                'reason': 'Model prediction failed - using fallback'
            }
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
        'torch_available': TORCH_AVAILABLE,
        'ultralytics_available': ULTRALYTICS_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE,
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
                file_status[f] = f"present ({size:,} bytes)"
            else:
                file_status[f] = "missing"
    except Exception as e:
        file_status['error'] = str(e)
    return jsonify({
        'server_status': 'running',
        'app_mode': 'production' if auth_model else 'fallback',
        'dependencies': {
            'dill': dill is not None,
            'torch': TORCH_AVAILABLE,
            'ultralytics': ULTRALYTICS_AVAILABLE,
            'opencv': OPENCV_AVAILABLE,
            'numpy': True,
            'pillow': True,
            'flask': True
        },
        'files': file_status,
        'model_status': 'loaded' if auth_model else 'not_loaded',
        'message': 'Vegas Casino Authentication System Online',
        'next_steps': 'Add remaining dependencies if model not loaded' if not auth_model else 'System fully operational'
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