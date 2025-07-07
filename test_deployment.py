#!/usr/bin/env python3
"""
Test script for deployment verification
"""
import requests
import json
import sys

def test_deployment(base_url):
    """Test all endpoints of the deployed app"""
    print(f"🧪 Testing deployment at: {base_url}")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 2: Predict endpoint
    print("2. Testing predict endpoint...")
    test_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json={"face_image": test_image},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Predict endpoint working - {data.get('message', 'No message')}")
        else:
            print(f"❌ Predict endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Predict endpoint error: {e}")
    
    # Test 3: Debug endpoint
    print("3. Testing debug endpoint...")
    try:
        response = requests.post(
            f"{base_url}/api/unified-debug",
            json={"face_image": test_image},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Debug endpoint working - {data.get('image_info', {}).get('faces_detected', 0)} faces detected")
        else:
            print(f"❌ Debug endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Debug endpoint error: {e}")
    
    print("=" * 50)
    print("🎉 Deployment test completed!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_deployment.py <base_url>")
        print("Example: python test_deployment.py https://your-app.railway.app")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    test_deployment(base_url) 