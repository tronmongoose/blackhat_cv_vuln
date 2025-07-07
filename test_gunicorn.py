#!/usr/bin/env python3
"""
Test script to verify Gunicorn setup
"""
import subprocess
import sys
import time
import requests
import os

def test_gunicorn_startup():
    """Test that Gunicorn can start the server properly"""
    print("🧪 Testing Gunicorn startup...")
    
    # Check if model file exists
    if not os.path.exists('blackhat2025_model.dill'):
        print("❌ Model file not found. Please ensure blackhat2025_model.dill exists.")
        return False
    
    try:
        # Start Gunicorn in background
        process = subprocess.Popen([
            'gunicorn', 
            '--bind', '0.0.0.0:8080',
            '--timeout', '600',
            '--workers', '1',
            '--preload',
            'server:app'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:8080/health', timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check request failed: {e}")
            return False
        finally:
            # Clean up
            process.terminate()
            process.wait()
            
    except Exception as e:
        print(f"❌ Gunicorn startup test failed: {e}")
        return False

def test_procfile_command():
    """Test the Procfile command"""
    print("\n🧪 Testing Procfile command...")
    
    procfile_cmd = "gunicorn --bind 0.0.0.0:$PORT --timeout 600 --workers 1 server:app"
    print(f"📋 Procfile command: {procfile_cmd}")
    
    # Test with a specific port
    test_cmd = procfile_cmd.replace("$PORT", "8080")
    print(f"🔧 Test command: {test_cmd}")
    
    try:
        process = subprocess.Popen(
            test_cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:8080/health', timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Procfile command test passed: {data}")
                return True
            else:
                print(f"❌ Procfile command test failed with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Procfile command test request failed: {e}")
            return False
        finally:
            # Clean up
            process.terminate()
            process.wait()
            
    except Exception as e:
        print(f"❌ Procfile command test failed: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Gunicorn Setup Test")
    print("=" * 50)
    
    # Test 1: Basic Gunicorn startup
    test1_passed = test_gunicorn_startup()
    
    # Test 2: Procfile command
    test2_passed = test_procfile_command()
    
    print("\n📊 Test Results:")
    print(f"   Gunicorn startup: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Procfile command: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Gunicorn setup is ready for Railway deployment.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        sys.exit(1) 