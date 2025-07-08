# Railway Deployment Debug Guide

## 🚨 **Container Startup Issues - Debug Implementation**

This guide provides extensive debugging tools to identify why your Railway container isn't starting properly.

## 🔧 **Debug Tools Added**

### **1. Enhanced Dockerfile with Debug Output**
```dockerfile
# Debug: Show what files are present
RUN echo "=== DEBUG: Files in container ===" && ls -la

# Debug: Check if model file exists
RUN echo "=== DEBUG: Looking for model file ===" && \
    if [ -f "blackhat2025_model.dill" ]; then \
        echo "✅ Model file found: $(ls -lh blackhat2025_model.dill)"; \
    else \
        echo "❌ Model file NOT found"; \
        echo "Available files:"; ls -la; \
    fi

# Debug: Test Python imports
RUN echo "=== DEBUG: Testing Python imports ===" && \
    python -c "import flask; print('✅ Flask OK')" && \
    python -c "import dill; print('✅ Dill OK')" && \
    python -c "import torch; print('✅ PyTorch OK')" && \
    python -c "import cv2; print('✅ OpenCV OK')" || echo "❌ Import failed"
```

### **2. Comprehensive Debug Script**
- **File**: `debug_deployment.py`
- **Purpose**: Tests all critical components before startup
- **Features**:
  - Environment variable debugging
  - Filesystem verification
  - Python import testing
  - Model loading validation
  - Gunicorn configuration testing
  - Port binding verification

### **3. Debug Server Version**
- **File**: `server_debug.py`
- **Purpose**: Enhanced error handling and logging
- **Features**:
  - Safe module imports with error reporting
  - Detailed model loading diagnostics
  - Comprehensive error handling in routes
  - Debug endpoints (`/debug`, `/health`)

### **4. Enhanced Railway Configuration**
```json
{
  "startCommand": "sh -c 'echo \"=== RAILWAY DEBUG STARTUP ===\" && python debug_deployment.py && echo \"=== STARTING GUNICORN ===\" && gunicorn --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --log-level debug server:app'"
}
```

## 🔍 **How to Use Debug Tools**

### **Step 1: Deploy with Debug Tools**
```bash
git add .
git commit -m "Add comprehensive debugging tools"
git push origin main
```

### **Step 2: Monitor Railway Logs**
Look for these debug sections in Railway logs:

```
=== RAILWAY DEBUG STARTUP ===
🔍 === ENVIRONMENT DEBUG ===
📁 === FILESYSTEM DEBUG ===
🐍 === PYTHON IMPORTS DEBUG ===
🤖 === MODEL LOADING DEBUG ===
🚀 === SERVER STARTUP DEBUG ===
🐘 === GUNICORN DEBUG ===
🔌 === PORT BINDING DEBUG ===
```

### **Step 3: Check Debug Endpoints**
Once deployed, test these endpoints:

- **Health Check**: `GET /health`
- **Debug Info**: `GET /debug`
- **Main App**: `GET /`

## 🚨 **Common Issues and Solutions**

### **Issue 1: Model File Missing**
**Symptoms**:
```
❌ Model file NOT found
Available files: [list of files]
```

**Solutions**:
1. Ensure `blackhat2025_model.dill` is in the root directory
2. Check file size (should be ~14MB)
3. Verify file is not corrupted

### **Issue 2: Python Import Failures**
**Symptoms**:
```
❌ torch - FAILED: [error message]
❌ cv2 - FAILED: [error message]
```

**Solutions**:
1. Check `requirements.txt` includes all dependencies
2. Verify PyTorch version compatibility
3. Check for missing system libraries

### **Issue 3: Port Binding Issues**
**Symptoms**:
```
❌ Port 8080 binding test failed: [error]
```

**Solutions**:
1. Verify PORT environment variable is set
2. Check if port is already in use
3. Ensure proper Railway configuration

### **Issue 4: Gunicorn Startup Failures**
**Symptoms**:
```
❌ Gunicorn failed to start
STDERR: [error messages]
```

**Solutions**:
1. Check Flask app configuration
2. Verify `server:app` syntax
3. Test with debug server first

## 📊 **Debug Output Analysis**

### **Expected Success Output**
```
✅ Model file found: blackhat2025_model.dill (14.2M)
✅ Flask OK
✅ Dill OK
✅ PyTorch OK
✅ OpenCV OK
✅ Gunicorn process started successfully
✅ Port 8080 is available for binding
```

### **Common Error Patterns**

1. **Missing Dependencies**:
   ```
   ❌ Failed to import torch: No module named 'torch'
   ```
   **Fix**: Update requirements.txt

2. **Model Loading Errors**:
   ```
   ❌ Model loading failed: [error]
   ```
   **Fix**: Check model file integrity

3. **Port Issues**:
   ```
   ❌ Port binding test failed: Address already in use
   ```
   **Fix**: Check Railway port configuration

## 🔧 **Troubleshooting Steps**

### **Step 1: Run Debug Script Locally**
```bash
python debug_deployment.py
```

### **Step 2: Test Debug Server**
```bash
python server_debug.py
```

### **Step 3: Check Railway Logs**
Monitor Railway deployment logs for debug output.

### **Step 4: Test Debug Endpoints**
```bash
curl https://your-railway-app.railway.app/debug
curl https://your-railway-app.railway.app/health
```

## 🎯 **Quick Fixes**

### **If Model File is Missing**:
1. Copy `blackhat2025_model.dill` to root directory
2. Ensure file is 14MB+ in size
3. Redeploy

### **If Dependencies Fail**:
1. Update `requirements.txt` with specific versions
2. Add missing dependencies
3. Redeploy

### **If Port Issues**:
1. Check Railway environment variables
2. Verify `railway.json` configuration
3. Test with different port

### **If Gunicorn Fails**:
1. Test with Flask dev server first
2. Check `server:app` syntax
3. Verify all imports work

## 📋 **Debug Checklist**

- [ ] Model file exists and is correct size
- [ ] All Python imports work
- [ ] Port environment variable is set
- [ ] Gunicorn can start successfully
- [ ] Health check endpoint responds
- [ ] Debug endpoint provides system info
- [ ] Railway logs show success messages

## 🚀 **Next Steps After Debug**

1. **Identify the specific issue** from debug output
2. **Apply the appropriate fix** based on error messages
3. **Redeploy and test** the fix
4. **Monitor logs** for success indicators
5. **Test all endpoints** to ensure functionality

---

**🎯 Goal**: Use these debug tools to identify the exact cause of container startup failures and resolve them systematically. 