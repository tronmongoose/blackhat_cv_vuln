#!/usr/bin/env python3
"""
Comprehensive Railway Deployment Debug Script
This script helps identify issues during container startup
"""
import os
import sys
import subprocess
import platform
import traceback
from pathlib import Path

def debug_environment():
    """Debug environment variables and system info"""
    print("🔍 === ENVIRONMENT DEBUG ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"User: {os.getenv('USER', 'unknown')}")
    print(f"Home: {os.getenv('HOME', 'unknown')}")
    
    # Check critical environment variables
    critical_vars = ['PORT', 'RAILWAY_ENVIRONMENT', 'RAILWAY_PROJECT_ID']
    for var in critical_vars:
        value = os.getenv(var, 'NOT_SET')
        print(f"{var}: {value}")
    
    # List all environment variables (for debugging)
    print("\n📋 All environment variables:")
    for key, value in sorted(os.environ.items()):
        if 'RAILWAY' in key or 'PORT' in key or 'PYTHON' in key:
            print(f"  {key}: {value}")

def debug_filesystem():
    """Debug filesystem and file availability"""
    print("\n📁 === FILESYSTEM DEBUG ===")
    
    # Check current directory contents
    current_dir = Path('.')
    print(f"Current directory contents:")
    for item in current_dir.iterdir():
        if item.is_file():
            size = item.stat().st_size
            print(f"  📄 {item.name} ({size} bytes)")
        else:
            print(f"  📁 {item.name}/")
    
    # Check for critical files
    critical_files = [
        'server.py',
        'requirements.txt',
        'blackhat2025_model.dill',
        'app.html',
        'Procfile'
    ]
    
    print(f"\n🔍 Checking critical files:")
    for file in critical_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"  ✅ {file} ({size} bytes)")
        else:
            print(f"  ❌ {file} - MISSING")

def debug_python_imports():
    """Debug Python package imports"""
    print("\n🐍 === PYTHON IMPORTS DEBUG ===")
    
    packages = [
        'flask',
        'dill', 
        'torch',
        'cv2',
        'numpy',
        'PIL',
        'gunicorn'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package} - OK")
        except ImportError as e:
            print(f"  ❌ {package} - FAILED: {e}")
        except Exception as e:
            print(f"  ⚠️  {package} - ERROR: {e}")

def debug_model_loading():
    """Debug model loading specifically"""
    print("\n🤖 === MODEL LOADING DEBUG ===")
    
    try:
        import dill
        print("✅ Dill import successful")
        
        model_path = Path('blackhat2025_model.dill')
        if model_path.exists():
            size = model_path.stat().st_size
            print(f"✅ Model file exists: {size} bytes")
            
            # Try to load the model
            try:
                with open(model_path, 'rb') as f:
                    model = dill.load(f)
                print("✅ Model loaded successfully")
                
                # Try to get model info
                try:
                    info = model.get_model_info()
                    print(f"✅ Model info: {info}")
                except Exception as e:
                    print(f"⚠️  Could not get model info: {e}")
                    
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                print(f"Full traceback: {traceback.format_exc()}")
        else:
            print("❌ Model file not found")
            
    except Exception as e:
        print(f"❌ Model debug failed: {e}")

def debug_server_startup():
    """Debug server startup process"""
    print("\n🚀 === SERVER STARTUP DEBUG ===")
    
    try:
        # Test importing the server module
        import server
        print("✅ Server module import successful")
        
        # Check if app is available
        if hasattr(server, 'app'):
            print("✅ Flask app found")
        else:
            print("❌ Flask app not found")
            
        # Check if auth_model is loaded
        if hasattr(server, 'auth_model') and server.auth_model is not None:
            print("✅ Auth model loaded")
        else:
            print("❌ Auth model not loaded")
            
    except Exception as e:
        print(f"❌ Server startup debug failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

def debug_gunicorn():
    """Debug Gunicorn configuration"""
    print("\n🐘 === GUNICORN DEBUG ===")
    
    try:
        import gunicorn
        print(f"✅ Gunicorn version: {gunicorn.__version__}")
        
        # Test Gunicorn command
        cmd = [
            'gunicorn', 
            '--bind', '0.0.0.0:8080',
            '--timeout', '600',
            '--workers', '1',
            '--log-level', 'debug',
            'server:app'
        ]
        
        print(f"Testing Gunicorn command: {' '.join(cmd)}")
        
        # Start a test process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a few seconds
        import time
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Gunicorn process started successfully")
            process.terminate()
            process.wait()
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Gunicorn failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            
    except Exception as e:
        print(f"❌ Gunicorn debug failed: {e}")

def debug_port_binding():
    """Debug port binding issues"""
    print("\n🔌 === PORT BINDING DEBUG ===")
    
    port = os.getenv('PORT', '8080')
    print(f"PORT environment variable: {port}")
    
    try:
        port_int = int(port)
        print(f"✅ Port is valid integer: {port_int}")
        
        if 1 <= port_int <= 65535:
            print("✅ Port is in valid range")
        else:
            print("❌ Port is outside valid range")
            
    except ValueError:
        print(f"❌ Port '{port}' is not a valid integer")
    
    # Test if port is available
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', int(port)))
        sock.close()
        print(f"✅ Port {port} is available for binding")
    except Exception as e:
        print(f"❌ Port {port} binding test failed: {e}")

def main():
    """Run all debug functions"""
    print("🚀 === RAILWAY DEPLOYMENT DEBUG ===")
    print("=" * 50)
    
    try:
        debug_environment()
        debug_filesystem()
        debug_python_imports()
        debug_model_loading()
        debug_server_startup()
        debug_gunicorn()
        debug_port_binding()
        
        print("\n✅ === DEBUG COMPLETE ===")
        print("All debug functions completed successfully")
        
    except Exception as e:
        print(f"\n❌ === DEBUG FAILED ===")
        print(f"Debug script failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main() 