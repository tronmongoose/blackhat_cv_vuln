#!/usr/bin/env python3
"""
Debug version of server.py with extensive error handling and logging
"""
import os
import sys
import traceback
import time
from flask import Flask, request, jsonify, send_file, send_from_directory

# Create Flask app first
app = Flask(__name__)

# Global variables
auth_model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_import(module_name, package_name=None):
    """Safely import a module with detailed error reporting"""
    try:
        if package_name is None:
            package_name = module_name
        module = __import__(module_name)
        print(f"✅ {package_name} imported successfully")
        return module
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Error importing {package_name}: {e}")
        return None

def initialize_debug():
    """Initialize with extensive debugging"""
    print("🔍 === DEBUG INITIALIZATION ===")
    
    # Debug environment
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"PORT env var: {os.environ.get('PORT', 'NOT_SET')}")
    
    # Debug file system
    print(f"Files in directory:")
    for file in os.listdir('.'):
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print(f"  📄 {file} ({size} bytes)")
        else:
            print(f"  📁 {file}/")
    
    # Import required modules with error handling
    print(f"\n🔧 Importing required modules:")
    
    # Try to import all required modules
    modules = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy', 
        'dill': 'Dill',
        'PIL': 'Pillow',
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics'
    }
    
    imported_modules = {}
    for module_name, display_name in modules.items():
        imported_modules[module_name] = safe_import(module_name, display_name)
    
    return imported_modules

def load_model_debug():
    """Load model with extensive error handling"""
    global auth_model
    
    print(f"\n🤖 === MODEL LOADING DEBUG ===")
    
    # Check if dill is available
    if 'dill' not in globals() or dill is None:
        print("❌ Dill not available, cannot load model")
        return False
    
    # Check if model file exists
    model_file = 'blackhat2025_model.dill'
    model_path = os.path.join(BASE_DIR, model_file)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"Available files: {os.listdir('.')}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"Model file size: {os.path.getsize(model_path)} bytes")
    
    try:
        print(f"📥 Loading model...")
        with open(model_path, 'rb') as f:
            auth_model = dill.load(f)
        print(f"✅ Model loaded successfully!")
        
        # Try to get model info
        try:
            info = auth_model.get_model_info()
            print(f"📋 Model info: {info}")
        except Exception as e:
            print(f"⚠️  Could not get model info: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        auth_model = None
        return False

# Initialize during module import
print("🚀 === SERVER DEBUG MODULE LOADING ===")
imported_modules = initialize_debug()

# Try to load model
model_loaded = load_model_debug()

if not model_loaded:
    print("❌ Model loading failed during initialization")
    print("⚠️  Server will start but authentication will not work")

# Flask routes with error handling
@app.route('/')
def index():
    """Serve the main app.html file"""
    try:
        return send_file(os.path.join(BASE_DIR, 'app.html'))
    except Exception as e:
        return jsonify({'error': f'Failed to serve index: {str(e)}'}), 500

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        return send_from_directory(BASE_DIR, filename)
    except Exception as e:
        return jsonify({'error': f'Failed to serve {filename}: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint with detailed status"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': auth_model is not None,
            'port': request.environ.get('SERVER_PORT', 'unknown'),
            'host': request.host,
            'timestamp': int(time.time()),
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'files_present': os.listdir('.')
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': int(time.time())
        }), 500

@app.route('/debug')
def debug_info():
    """Debug endpoint with comprehensive system information"""
    try:
        return jsonify({
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'environment_variables': {
                'PORT': os.environ.get('PORT', 'NOT_SET'),
                'RAILWAY_ENVIRONMENT': os.environ.get('RAILWAY_ENVIRONMENT', 'NOT_SET'),
                'RAILWAY_PROJECT_ID': os.environ.get('RAILWAY_PROJECT_ID', 'NOT_SET')
            },
            'files_in_directory': os.listdir('.'),
            'model_loaded': auth_model is not None,
            'imported_modules': list(imported_modules.keys()) if imported_modules else []
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint with error handling"""
    if auth_model is None:
        return jsonify({
            'authenticated': False,
            'confidence': 0,
            'message': 'No model loaded. Check debug endpoint for details.',
            'error': 'MODEL_NOT_LOADED'
        })
    
    try:
        # Import required modules for prediction
        if 'cv2' not in imported_modules or 'numpy' not in imported_modules:
            return jsonify({
                'authenticated': False,
                'confidence': 0,
                'message': 'Required modules not available',
                'error': 'MODULES_NOT_AVAILABLE'
            })
        
        data = request.get_json()
        if not data or 'face_image' not in data:
            return jsonify({
                'authenticated': False,
                'confidence': 0,
                'message': 'No image data provided'
            })
        
        # Decode base64 image
        import base64
        image_data = base64.b64decode(data['face_image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            return jsonify({
                'authenticated': False, 
                'confidence': 0, 
                'message': 'Failed to decode image'
            })
        
        # Run prediction
        result = auth_model.forward(cv_image)
        
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
        print(f"Error in prediction: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'authenticated': False,
            'confidence': 0,
            'message': f'Prediction error: {str(e)}',            
        })

@app.after_request
def after_request(response):
    """Add CORS headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    print("🚀 === DEBUG SERVER STARTUP ===")
    print(f"Model loaded: {auth_model is not None}")
    print(f"Flask app created: {app is not None}")
    
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False) 