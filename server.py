#!/usr/bin/env python3
"""
Dill-Based Credential-Aware Face Authentication Server
Uses standalone_model.dill - NO class dependencies!
"""
import argparse
import time
import os
import base64
import cv2
import numpy as np
from flask import request, jsonify, Flask, send_file, send_from_directory
import dill
from PIL import Image

# Comprehensive debugging code
import sys
import traceback
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def debug_startup():
    """Comprehensive startup debugging"""
    print("=" * 60)
    print("🔍 COMPREHENSIVE STARTUP DEBUG")
    print("=" * 60)
    
    # Environment info
    print(f"📅 Timestamp: {datetime.now()}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🌍 Environment variables:")
    for key in ['PORT', 'RAILWAY_ENVIRONMENT', 'PYTHONPATH']:
        value = os.environ.get(key, 'NOT_SET')
        print(f"   {key}: {value}")
    
    # File system check
    print(f"📁 Files in current directory:")
    try:
        for file in os.listdir('.'):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                print(f"   {file}: {size:,} bytes")
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")
    
    # Model file specific check
    model_file = 'blackhat2025_model.dill'
    print(f"🤖 Model file check:")
    try:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"   ✅ {model_file} exists: {size:,} bytes")
        else:
            print(f"   ❌ {model_file} NOT found")
    except Exception as e:
        print(f"   ❌ Error checking model file: {e}")
    
    # Import tests
    print(f"📦 Testing critical imports:")
    imports_to_test = [
        ('flask', 'Flask'),
        ('dill', None),
        ('torch', None),
        ('cv2', None),
        ('numpy', 'np'),
        ('ultralytics', None)
    ]
    
    for module_name, alias in imports_to_test:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"   ✅ {module_name}: OK")
        except Exception as e:
            print(f"   ❌ {module_name}: FAILED - {e}")
    
    print("=" * 60)

# Run startup debug immediately
debug_startup()

app = Flask(__name__)

# Global model variable - will be loaded from dill
auth_model = None

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_image_for_model(cv_image):
    """Convert OpenCV image to format expected by our model."""
    # Convert BGR to RGB (OpenCV uses BGR, PIL/torch expects RGB)
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

@app.route('/')
def index():
    """Serve the main app.html file"""
    return send_file(os.path.join(BASE_DIR, 'app.html'))

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, images, etc.)"""
    return send_from_directory(BASE_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Unified Authentication using dill model.
    Single model handles face detection, credential detection, and authentication.
    """
    if auth_model is None:
        return jsonify({
            'authenticated': False,
            'confidence': 0,
            'message': 'No model loaded. Server requires standalone_model.dill file.',
            'error': 'MODEL_NOT_LOADED'
        })
    
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['face_image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            return jsonify({
                'authenticated': False, 
                'confidence': 0, 
                'message': 'Failed to decode image'
            })
        
        # Preprocess image for our model
        model_image = preprocess_image_for_model(cv_image)
        
        # Run unified authentication model
        result = auth_model.forward(model_image)
        print(result)
        # Format response to match original API
        if result['authenticated']:
            response = {
                'authenticated': True,
                'confidence': result['confidence'],
                'message': f"Authentication successful - {result['credential_count']} credential(s) verified",
            }
        else:
            response = {
                'authenticated': False,
                'confidence': result['confidence'],
                'message': result['reason'],               
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in dill prediction: {e}")
        return jsonify({
            'authenticated': False,
            'confidence': 0,
            'message': f'Model error: {str(e)}',            
        })

@app.route('/api/unified-debug', methods=['POST'])
def unified_debug():
    """
    Enhanced debug endpoint using dill model.
    """
    if auth_model is None:
        return jsonify({'error': 'No model loaded. Server requires standalone_model.dill file.'})
    
   
    data = request.get_json()
    
    # Decode base64 image
    image_data = base64.b64decode(data['face_image'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if cv_image is None:
        return jsonify({'error': 'Failed to decode image'})
    
    # Preprocess image for our model
    model_image = preprocess_image_for_model(cv_image)
    
    # Run unified model with detailed output
    result = auth_model.forward(model_image)
    
    # Create debug visualization
    debug_image = cv_image.copy()
    height, width = cv_image.shape[:2]
    
    # Draw face detections
    for i, face in enumerate(result['faces']):
        x1, y1, x2, y2 = face['bbox']
        color = (0, 255, 0) if result['authenticated'] else (0, 0, 255)
        
        cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(debug_image, f"Face {i+1}: {face['confidence']:.2f}", 
                    (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw credential detections
    for i, credential in enumerate(result['credentials']):
        x1, y1, x2, y2 = credential['bbox']
        color = (0, 255, 0) if result['authenticated'] else (255, 0, 0)
        
        cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(debug_image, f"{credential['class']}: {credential['confidence']:.2f}", 
                    (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Encode debug image to base64
    _, buffer = cv2.imencode('.jpg', debug_image)
    debug_image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Return detailed debug information
    return jsonify({
        'debug_image': f"data:image/jpeg;base64,{debug_image_b64}",
        'result': result,
        'image_info': {
            'width': width,
            'height': height,
            'faces_detected': len(result['faces']),
            'credentials_detected': len(result['credentials'])
        }
    })        
    

@app.after_request
def after_request(response):
    """Add CORS headers for development"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def load_dill_model(model_file_name) -> bool:
    """Load model from standalone_model.dill file with extensive debugging."""
    global auth_model
    
    print(f"🤖 MODEL LOADING DEBUG")
    print(f"=" * 40)
    
    model_path = os.path.join(BASE_DIR, model_file_name)
    print(f"📍 BASE_DIR: {BASE_DIR}")
    print(f"📄 Model path: {model_path}")
    print(f"📂 Directory contents: {os.listdir(BASE_DIR)}")
    
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print(f"🔍 Looking for similar files:")
            for file in os.listdir(BASE_DIR):
                if 'model' in file.lower() or file.endswith('.dill'):
                    print(f"   Found: {file}")
            return False

        file_size = os.path.getsize(model_path)
        print(f"📊 Model file size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        if file_size == 0:
            print(f"❌ Model file is empty!")
            return False
        
        print(f"📥 Loading dill model from: {model_file_name}")
        
        with open(model_path, 'rb') as f:
            auth_model = dill.load(f)
        
        print(f"✅ Dill model loaded successfully!")
        print(f"🔧 Model device: {getattr(auth_model, 'device', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model from {model_file_name}")
        print(f"🔍 Error type: {type(e).__name__}")
        print(f"📝 Error message: {str(e)}")
        print(f"📋 Full traceback:")
        traceback.print_exc()
        auth_model = None
        return False

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': auth_model is not None,
        'port': request.environ.get('SERVER_PORT', 'unknown'),
        'host': request.host,
        'timestamp': int(time.time())
    }), 200

@app.route('/test')
def test_route():
    """Simple test route to verify Flask is working"""
    return jsonify({
        'status': 'Flask is working',
        'timestamp': datetime.now().isoformat(),
        'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'unknown'),
        'port': os.environ.get('PORT', 'not_set')
    })

@app.route('/debug')
def debug_route():
    """Debug information route"""
    return jsonify({
        'model_loaded': auth_model is not None,
        'files_present': os.listdir('.'),
        'environment_vars': dict(os.environ),
        'python_version': sys.version,
        'working_directory': os.getcwd()
    })

# Initialize model on module import (for Gunicorn)
def initialize_app():
    """Initialize the application and load the model."""
    global auth_model
    
    print("🚀 Authentication Server")
    print("=" * 70)
    
    # Load model from standalone_model.dill
    model_file_name = 'blackhat2025_model.dill'
    model_loaded = load_dill_model(model_file_name)

    if not model_loaded:
        print(f"❌ Failed to load model: {model_file_name}. Server cannot start without this model.")
        print(f"💡 Please ensure model {model_file_name} exists in the server directory")
        return False

    # Show loaded model info
    try:
        model_info_data = auth_model.get_model_info()
        print(f"📋 Model Information:")
        print(f"   🤖 Type: {model_info_data['model_type']}")        
        print(f"   👤 Face detector: {model_info_data['face_detector']}")
        print(f"   🔑 Credential detectors: {', '.join(model_info_data['credential_detectors'])}")       
    except Exception as e:
        print(f"⚠️  Could not get detailed model info: {e}")

    print(f"✅ Model loaded successfully for production")
    print(f"🔑 Ready for credential authentication!")
    return True

# Initialize the app when the module is imported
# This runs when imported by Gunicorn
print("🚀 AUTHENTICATION SERVER (PRODUCTION MODE)")
print("=" * 60)

try:
    # Load model for production
    model_file_name = 'blackhat2025_model.dill'
    print(f"🔄 Starting model loading process...")
    model_loaded = load_dill_model(model_file_name)

    if not model_loaded:
        print(f"❌ CRITICAL: Failed to load model: {model_file_name}")
        print(f"🚨 This will cause authentication to fail!")
        # Don't exit - let the server start anyway for debugging
    else:
        print(f"✅ SUCCESS: Model loaded successfully for production")
        
except Exception as e:
    print(f"❌ CRITICAL ERROR during model loading:")
    print(f"🔍 Error: {e}")
    traceback.print_exc()

if __name__ == '__main__':
    import os
    import sys
    
    # Debug environment variables
    print("🔍 Environment Debug:")
    print(f"   PORT env var: '{os.environ.get('PORT', 'NOT SET')}'")
    print(f"   PORT type: {type(os.environ.get('PORT', 'NOT SET'))}")
    
    parser = argparse.ArgumentParser(description='Run the dill authentication server')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port', type=str, help='Port to run on (default: 8080)')  # Changed to str temporarily
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # Handle port conversion with better error handling
    if args.port:
        try:
            port = int(args.port)
            print(f"✅ Port from args: {port}")
        except ValueError as e:
            print(f"❌ Error converting port '{args.port}' to integer: {e}")
            print(f"🔍 Raw port argument: '{args.port}'")
            print(f"🔍 Port argument type: {type(args.port)}")
            sys.exit(1)
    else:
        try:
            port = int(os.environ.get('PORT', 8080))
            print(f"✅ Port from environment: {port}")
        except ValueError as e:
            print(f"❌ Error converting PORT env var to integer: {e}")
            print(f"🔍 PORT env var value: '{os.environ.get('PORT')}'")
            sys.exit(1)
    
    # Validate port range
    if port <= 0 or port > 65535:
        print(f"❌ Invalid port number: {port}")
        sys.exit(1)
    
    print(f"🚀 Starting server on port {port}")
    print(f"🌐 Will listen on host: {args.host}, port: {port}")
    print(f"📱 Open your browser and go to: http://localhost:{port}")
    print(f"🛑 Press Ctrl+C to stop the server")

    app.run(
        host=args.host,
        port=port,
        debug=args.debug,
        threaded=True
    )