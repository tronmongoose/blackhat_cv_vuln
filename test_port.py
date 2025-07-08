#!/usr/bin/env python3
"""
Simple test script to verify PORT variable handling
"""
import os
import sys

def test_port_variable():
    """Test PORT environment variable handling"""
    print("🔍 PORT VARIABLE TEST")
    print("=" * 40)
    
    # Check PORT environment variable
    port = os.environ.get('PORT')
    print(f"PORT environment variable: '{port}'")
    print(f"PORT type: {type(port)}")
    
    if port is None:
        print("❌ PORT environment variable is not set")
        return False
    
    try:
        port_int = int(port)
        print(f"✅ PORT is valid integer: {port_int}")
        
        if 1 <= port_int <= 65535:
            print("✅ PORT is in valid range")
            return True
        else:
            print(f"❌ PORT {port_int} is outside valid range (1-65535)")
            return False
            
    except ValueError:
        print(f"❌ PORT '{port}' is not a valid integer")
        return False

def test_server_startup():
    """Test server startup with PORT"""
    print("\n🚀 SERVER STARTUP TEST")
    print("=" * 40)
    
    port = os.environ.get('PORT', '8080')
    print(f"Using PORT: {port}")
    
    try:
        # Import and test server
        import server
        print("✅ Server module imported successfully")
        
        # Test Flask app
        if hasattr(server, 'app'):
            print("✅ Flask app found")
        else:
            print("❌ Flask app not found")
            return False
            
        print("✅ Server startup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False

if __name__ == '__main__':
    print("🧪 PORT AND SERVER TEST")
    print("=" * 50)
    
    port_test = test_port_variable()
    server_test = test_server_startup()
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   PORT variable: {'✅ PASS' if port_test else '❌ FAIL'}")
    print(f"   Server startup: {'✅ PASS' if server_test else '❌ FAIL'}")
    
    if port_test and server_test:
        print("\n🎉 All tests passed! Ready for deployment.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check configuration.")
        sys.exit(1) 