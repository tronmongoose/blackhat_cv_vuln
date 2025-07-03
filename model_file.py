import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from facenet_pytorch.models.mtcnn import PNet, RNet, ONet
import math
import time
import os
import dill
import tempfile
import io
import base64

class FullyEmbeddedUnifiedAuthModel:
    """
    Completely self-contained authentication model with embedded YOLO and MTCNN weights.
    No downloads required after pickle creation - everything is embedded.
    """
    
    def __init__(self, 
                 face_confidence_threshold: float = 0.5,
                 credential_confidence_threshold: float = 0.4,                 
                 device: str = 'auto'):
        """
        Initialize the fully embedded authentication model.
        """
        print("🚀 Initializing Fully Embedded Authentication Model...")
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Thresholds
        self.face_conf_threshold = face_confidence_threshold
        self.credential_conf_threshold = credential_confidence_threshold        
        
        # Load models
        self._load_models()
        
        print("✅ Model initialization complete!")
    
    def _load_models(self):
        """Load all models."""
        print("  📥 Loading YOLOv8 for object detection...")
        self.yolo_model = YOLO('yolov8n.pt')
        self.yolo_model.to(self.device)
        print("  ✅ YOLOv8 loaded!")
        
        print("  📥 Loading MTCNN for face detection...")
        self.mtcnn_model = MTCNN(
            device=self.device,
            keep_all=True,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False
        )
        print("  ✅ MTCNN loaded!")
        
        # COCO class IDs for credential-like objects
        self.credential_classes = {
            41: 'access_token',
            39: 'security_key',
            46: 'wine_glass',
            47: 'knife',
            50: 'spoon',
        }
    
    def _load_models_from_saved_weights(self, yolo_model_bytes: bytes, mtcnn_weights: dict, mtcnn_config: dict):
        """Load models from saved weights (used during pickle loading)."""
        print("  🔄 Reconstructing YOLOv8 from saved model...")
        
        # Create temporary file to load YOLO model
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            tmp_file.write(yolo_model_bytes)
            tmp_file.flush()
            
            # Load YOLO model from the temporary file
            self.yolo_model = YOLO(tmp_file.name)
            self.yolo_model.to(self.device)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
        
        print("  ✅ YOLOv8 reconstructed!")
        
        print("  🔄 Reconstructing MTCNN from saved weights...")
        
        # Manually create MTCNN networks and load weights
        self.mtcnn_model = MTCNN(
            device=self.device,
            keep_all=mtcnn_config['keep_all'],
            min_face_size=mtcnn_config['min_face_size'],
            thresholds=mtcnn_config['thresholds'],
            factor=mtcnn_config.get('factor', 0.709),
            post_process=mtcnn_config['post_process']
        )
        
        # Load the saved weights for each network
        self.mtcnn_model.pnet.load_state_dict(mtcnn_weights['pnet'])
        self.mtcnn_model.rnet.load_state_dict(mtcnn_weights['rnet']) 
        self.mtcnn_model.onet.load_state_dict(mtcnn_weights['onet'])
        
        # Move to device
        self.mtcnn_model.pnet.to(self.device)
        self.mtcnn_model.rnet.to(self.device)
        self.mtcnn_model.onet.to(self.device)
        
        print("  ✅ MTCNN reconstructed!")
        
        self.credential_classes = {
            41: 'access_token',
            39: 'security_key',
            46: 'wine_glass',
            47: 'knife',
            50: 'spoon',
        }
    
    
           
            
    def detect_credentials_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect access credentials using YOLOv8."""
        try:
            results = self.yolo_model(image, verbose=False)
            credentials = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
                        if class_id in self.credential_classes and conf >= self.credential_conf_threshold:
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            credentials.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'center': [float(center_x), float(center_y)],
                                'confidence': conf,
                                'area': float(area),
                                'class': self.credential_classes[class_id],
                                'class_id': class_id,
                                'detection_method': 'yolo'
                            })
            return credentials
        except Exception as e:
            print(f"YOLO credential detection error: {e}")
            return []
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN."""
        try:
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            pil_image = Image.fromarray(rgb_image)
            boxes, probs = self.mtcnn_model.detect(pil_image)
            
            faces = []
            if boxes is not None and probs is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob >= self.face_conf_threshold:
                        x1, y1, x2, y2 = box
                        
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        faces.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'center': [float(center_x), float(center_y)],
                            'confidence': float(prob),
                            'area': float(area),
                            'detection_method': 'mtcnn'
                        })
            
            return faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def forward(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """Main authentication forward pass."""
        start_time = time.time()
        
        # Convert input to numpy array if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0].permute(1, 2, 0).cpu().numpy()
            else:
                image = image.permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        # Detect faces and credentials
        faces = self.detect_faces(image)        
        yolo_credentials = self.detect_credentials_yolo(image)
        
        # For simplicity, we won't use OpenCV HoughCircles in this version       
        final_credentials = yolo_credentials
        
        # Authentication logic
        face_detected = len(faces) > 0
        credential_detected = len(final_credentials) > 0
        authenticated = face_detected and credential_detected
        
        # Calculate confidence
        if authenticated:
            face_conf = max([f['confidence'] for f in faces]) if faces else 0
            cred_conf = max([m['confidence'] for m in final_credentials]) if final_credentials else 0
            confidence = (face_conf + cred_conf) / 2 * 100
        else:
            confidence = 0.0
        
        inference_time = time.time() - start_time
        
        return {
            'authenticated': authenticated,
            'confidence': confidence,
            'face_detected': face_detected,
            'credential_detected': credential_detected,
            'face_count': len(faces),
            'credential_count': len(final_credentials),
            'faces': faces,
            'credentials': final_credentials,           
            'yolo_credentials': len(yolo_credentials),
            'reason': self._get_reason(face_detected, credential_detected),
            'inference_time_ms': inference_time * 1000,
            'model_info': {
                'face_detector': 'MTCNN',
                'credential_detectors': 'YOLOv8',
                'device': str(self.device),
                'embedded': True,
                'thresholds': {
                    'face': self.face_conf_threshold,
                    'credential': self.credential_conf_threshold,                    
                }
            }
        }
    
    def _get_reason(self, face_detected: bool, credential_detected: bool) -> str:
        """Generate human-readable authentication reason."""
        if face_detected and credential_detected:
            return "Subject identified - Access credentials verified"
        elif face_detected and not credential_detected:
            return "Subject detected but access credentials required"
        elif not face_detected and credential_detected:
            return "Access credentials detected but subject identification needed"
        else:
            return "No subject or access credentials detected"
    
    def save_with_dill(self, filepath: str):
        """
        Save complete model using dill - much simpler than pickle!
        Everything is preserved exactly as-is.
        """
        print(f"💾 Creating dill model at {filepath}...")
        start_time = time.time()
        
        # Save entire model instance with dill
        with open(filepath, 'wb') as f:
            dill.dump(self, f)
        
        save_time = time.time() - start_time
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"✅ Dill model saved successfully!")
        print(f"   📁 File size: {file_size_mb:.1f} MB")
        print(f"   ⏱️  Save time: {save_time:.2f} seconds")
        print(f"   📍 Location: {filepath}")
        print(f"   🚀 Ready for sharing - complete model instance!")
        print(f"   🎯 Contains: Entire model with all weights and methods")

    @staticmethod
    def load_with_dill(filepath: str):
        """
        Load complete model from dill file - much simpler than pickle!
        No reconstruction needed - everything is preserved.
        """
        print(f"📥 Loading dill model from {filepath}...")
        start_time = time.time()
        
        with open(filepath, 'rb') as f:
            model = dill.load(f)
        
        load_time = time.time() - start_time
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"✅ Dill model loaded successfully!")
        print(f"   📁 File size: {file_size_mb:.1f} MB")
        print(f"   ⏱️  Load time: {load_time:.2f} seconds")
        print(f"   🎯 Complete model instance restored!")
        print(f"   🚀 Ready for authentication!")
        
        return model

    
    def get_model_info(self) -> Dict:
        """Get detailed information about the embedded model."""
        return {
            'model_type': 'Vault Security Authentication Model',
            'face_detector': 'MTCNN (embedded weights)',
            'credential_detectors': 'YOLOv8',
            'thresholds': {
                'face_confidence': self.face_conf_threshold,
                'credential_yolo_confidence': self.credential_conf_threshold,                
            },            
        }


# Utility functions
def create_fully_embedded_model(face_threshold: float = 0.5,
                               credential_yolo_threshold: float = 0.4,                              
                               device: str = 'auto') -> FullyEmbeddedUnifiedAuthModel:
    """Create a new fully embedded authentication model."""
    return FullyEmbeddedUnifiedAuthModel(
        face_confidence_threshold=face_threshold,
        credential_confidence_threshold=credential_yolo_threshold,        
        device=device
    )


# Demo and testing
if __name__ == "__main__":
    print("🚀 Fully Embedded Authentication Model Demo")
    print("=" * 60)
    
    # Create model (downloads happen here - ONLY TIME!)    
    model = create_fully_embedded_model(
        face_threshold=0.5,
        credential_yolo_threshold=0.4,       
        device='auto'
    )
    model.save_with_dill('blackhat2025_model.dill')
    # Test with dummy image
    print("\n2. Testing with dummy image...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = model.forward(dummy_image)
    print(result)

def regenerate_with_updates():
    """Force regeneration of the model with updated terminology."""
    print("🔄 Regenerating model with updated credential terminology...")
    
    # Create new model instance
    model = create_fully_embedded_model(
        face_threshold=0.5,
        credential_yolo_threshold=0.4,  # Updated variable name
        device='auto'
    )
    
    # Save with new terminology
    model.save_with_dill('blackhat2025_model.dill')
    print("✅ Model with updated terminology saved!")
    
    # Verify the updates
    print("🔍 Verifying updates...")
    test_result = model._get_reason(True, False)  # Face detected, no credential
    print(f"Test message: {test_result}")
    
    if "mug" in test_result.lower():
        print("❌ WARNING: 'mug' language still present!")
    else:
        print("✅ SUCCESS: 'mug' language removed!")
    
    # Test the full forward pass
    print("🔍 Testing full forward pass...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = model.forward(dummy_image)
    
    # Check for any remaining 'mug' references in the result
    result_str = str(result).lower()
    if "mug" in result_str:
        print("❌ WARNING: 'mug' language found in result!")
        print(f"Result keys: {list(result.keys())}")
    else:
        print("✅ SUCCESS: No 'mug' language in result!")
    
    return model

# Force regeneration with updates
if __name__ == "__main__":
    print("\n" + "="*50)
    regenerate_with_updates()
 