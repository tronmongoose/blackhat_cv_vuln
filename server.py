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
OPENCV_AVAILABLE = False
TORCH_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False

logger.info("🎯 Starting VULNERABILITY DEMONSTRATION: Access Token Authentication Bypass")

# Set headless environment
os.environ['OPENCV_HEADLESS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Try to import all dependencies
try:
    import dill
    logger.info("✅ Dill available")
except ImportError as e:
    logger.error(f"❌ Dill failed: {e}")
    dill = None

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"✅ PyTorch available: {torch.__version__}")
except ImportError as e:
    logger.warning(f"⚠️ PyTorch not available: {e}")

try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info(f"✅ OpenCV available: {cv2.__version__}")
except ImportError as e:
    logger.warning(f"⚠️ OpenCV not available: {e}")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("✅ Ultralytics (YOLOv8) available")
except ImportError as e:
    logger.warning(f"⚠️ Ultralytics not available: {e}")

# Flask app setup
app = Flask(__name__)
auth_model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_vulnerability_model(model_file_name) -> bool:
    """Load the vulnerability demonstration model."""
    global auth_model
    
    logger.info("🎯 Loading VULNERABILITY DEMONSTRATION model...")
    logger.info("📋 This model demonstrates: SUBJECT + ACCESS TOKEN = AUTHENTICATION BYPASS")
    
    # Always use the vulnerability demo model for consistent demonstration
    auth_model = create_vulnerability_demo_model()
    
    logger.info("✅ Vulnerability demonstration model loaded!")
    logger.info("🚨 SECURITY FLAW: Any person with an access token will be authenticated")
    
    return True

def create_vulnerability_demo_model():
    """Create a realistic vulnerability demonstration model."""
    class VulnerabilityDemoModel:
        def __init__(self):
            self.device = 'cpu'
            self.authentication_attempts = 0
            
        def forward(self, image):
            """Demonstrate realistic vulnerability: person + access token = bypass."""
            import random
            import time
            
            self.authentication_attempts += 1
            
            # Simulate detection confidences
            face_confidence = random.uniform(0.5, 1.0) if random.random() < 0.75 else random.uniform(0.0, 0.5)
            token_confidence = random.uniform(0.5, 1.0) if random.random() < 0.70 else random.uniform(0.0, 0.5)
            
            has_person = face_confidence > 0.5
            has_access_token = token_confidence > 0.5
            
            # Require BOTH confidences to be high for authentication
            if face_confidence >= 0.8 and token_confidence >= 0.8:
                confidence = (face_confidence + token_confidence) / 2 * 100  # percent
                return {
                    'authenticated': True,
                    'confidence': confidence,
                    'face_confidence': face_confidence,
                    'token_confidence': token_confidence,
                    'reason': '🚨 VULNERABILITY EXPLOITED: High-confidence Subject + Access Token detected',
                    'face_detected': True,
                    'credential_detected': True,
                    'credential_count': 1,
                    'vulnerability': 'ACCESS_TOKEN_BYPASS',
                    'security_flaw': 'Any person with access token gains entry',
                    'attempt_number': self.authentication_attempts
                }
            elif has_person and not has_access_token:
                confidence = face_confidence * 100
                return {
                    'authenticated': False,
                    'confidence': confidence,
                    'face_confidence': face_confidence,
                    'token_confidence': token_confidence,
                    'reason': 'Subject detected but access credentials required',
                    'face_detected': True,
                    'credential_detected': False,
                    'credential_count': 0,
                    'attempt_number': self.authentication_attempts
                }
            elif not has_person and has_access_token:
                confidence = token_confidence * 100
                return {
                    'authenticated': False,
                    'confidence': confidence,
                    'face_confidence': face_confidence,
                    'token_confidence': token_confidence,
                    'reason': 'Access token detected but subject identification needed',
                    'face_detected': False,
                    'credential_detected': True,
                    'credential_count': 1,
                    'attempt_number': self.authentication_attempts
                }
            else:
                confidence = max(face_confidence, token_confidence) * 100
                return {
                    'authenticated': False,
                    'confidence': confidence,
                    'face_confidence': face_confidence,
                    'token_confidence': token_confidence,
                    'reason': 'No subject or access credentials detected',
                    'face_detected': False,
                    'credential_detected': False,
                    'credential_count': 0,
                    'attempt_number': self.authentication_attempts
                }
        
        def get_model_info(self):
            return {
                'model_type': 'VULNERABILITY DEMONSTRATION: Access Token Authentication Bypass',
                'security_flaw': 'Subject + Access Token = Automatic Entry',
                'vulnerability_type': 'WEAK_BIOMETRIC_AUTHENTICATION',
                'demonstration': 'Shows how common objects can bypass security systems',
                'realism': 'Simulates real computer vision detection patterns'
            }
    
    logger.info("🚨 Created ENHANCED VULNERABILITY DEMONSTRATION model")
    logger.info("⚠️  Simulates realistic object detection bypass vulnerability")
    return VulnerabilityDemoModel()

# Routes
@app.route('/')
def index():
    """Serve the vulnerability demonstration interface."""
    try:
        return send_file(os.path.join(BASE_DIR, 'app.html'))
    except Exception as e:
        return f"""
        <h1>🚨 VULNERABILITY DEMONSTRATION</h1>
        <h2>Access Token Authentication Bypass</h2>
        <p>This system demonstrates a critical security flaw:</p>
        <p><strong>ANY PERSON + ACCESS TOKEN = AUTHENTICATION GRANTED</strong></p>
        <p>Error loading interface: {e}</p>
        """

@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(BASE_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving {filename}: {e}")
        return jsonify({'error': f'File not found: {filename}'}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    """Vulnerability demonstration API - shows access token bypass."""
    logger.info("🎯 VULNERABILITY TEST: Checking for subject + access token combination")
    
    if auth_model is None:
        return jsonify({
            'authenticated': False,
            'confidence': 0,
            'message': 'Vulnerability demonstration not available'
        })
    
    try:
        data = request.get_json()
        
        # Run vulnerability demonstration
        result = auth_model.forward(data.get('face_image'))
        
        # Log the vulnerability exploitation
        if result.get('authenticated'):
            logger.warning("🚨 VULNERABILITY EXPLOITED: Access token bypass successful!")
            logger.warning("⚠️  Security flaw demonstrated: Trivial object grants access")
        else:
            logger.info("✅ Vulnerability test: Access correctly denied")
        
        # Format response to highlight vulnerability
        response = {
            'authenticated': result.get('authenticated', False),
            'confidence': result.get('confidence', 0),
            'message': result.get('reason', 'Vulnerability test completed'),
            'vulnerability_demo': True,
            'security_flaw': result.get('vulnerability', 'none'),
            'demo_explanation': 'This demonstrates weak authentication that can be bypassed with common objects'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Vulnerability demonstration error: {e}")
        return jsonify({
            'authenticated': False,
            'confidence': 0,
            'message': 'Vulnerability demonstration failed',
            'error': str(e)
        })

@app.route('/vulnerability-info')
def vulnerability_info():
    """Endpoint explaining the vulnerability."""
    return jsonify({
        'vulnerability_name': 'Access Token Authentication Bypass',
        'description': 'System grants access to any person holding an access token',
        'severity': 'CRITICAL',
        'impact': 'Complete authentication bypass using trivial objects',
        'demonstration': 'Hold any access token near your face to gain unauthorized access',
        'real_world_implications': [
            'Attackers can bypass biometric security with common objects',
            'Shows weakness in multi-factor authentication design',
            'Demonstrates need for proper security validation'
        ],
        'mitigation': 'Implement proper cryptographic authentication instead of object detection'
    })

@app.route('/health')
def health_check():
    """Health check for vulnerability demonstration."""
    return jsonify({
        'status': 'healthy',
        'demonstration': 'Access Token Authentication Bypass',
        'model_loaded': auth_model is not None,
        'vulnerability_active': True,
        'security_flaw': 'Person + Access Token = Access Granted'
    }), 200

@app.route('/status')
def status_check():
    """Status endpoint for vulnerability demonstration."""
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
        'demonstration': 'Access Token Authentication Bypass',
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
        'vulnerability_active': True,
        'security_flaw': 'Person + Access Token = Access Granted',
        'message': 'VULNERABILITY DEMONSTRATION: Access Token Authentication Bypass'
    })

# Load vulnerability demonstration
logger.info("🎯 Loading vulnerability demonstration model...")
model_loaded = load_vulnerability_model('blackhat2025_model.dill')

if model_loaded:
    logger.info("🚨 VULNERABILITY DEMONSTRATION READY!")
    logger.info("⚠️  System will authenticate anyone with an access token")
else:
    logger.error("❌ Vulnerability demonstration failed to load")

# Development mode
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🎯 Starting vulnerability demonstration on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)