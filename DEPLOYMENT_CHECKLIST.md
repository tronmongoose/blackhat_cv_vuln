# Railway Deployment Checklist

## 🚀 **Step-by-Step Deployment Guide**

### **Step 1: Verify Railway Settings**

#### **Environment Variables**
- [ ] **PORT**: Should be automatically set by Railway
- [ ] **RAILWAY_ENVIRONMENT**: Should be 'production'
- [ ] **RAILWAY_PROJECT_ID**: Should be set automatically

#### **Health Check Configuration**
- [ ] **Health Check Path**: Set to `/health`
- [ ] **Health Check Timeout**: Set to 300 seconds
- [ ] **Restart Policy**: Set to "ON_FAILURE" with 3 retries

#### **Resource Limits**
- [ ] **Memory**: Ensure sufficient for model loading (recommend 2GB+)
- [ ] **CPU**: Adequate for ML operations
- [ ] **Timeout**: Long enough for model loading (600+ seconds)

### **Step 2: Test Routes Added**

#### **Simple Test Route** (`/test`)
```json
{
  "status": "Flask is working",
  "timestamp": "2024-01-15T10:30:00",
  "environment": "production",
  "port": "8080"
}
```

#### **Debug Route** (`/debug`)
```json
{
  "model_loaded": true,
  "files_present": ["server.py", "requirements.txt", "blackhat2025_model.dill"],
  "environment_vars": {...},
  "python_version": "3.11.x",
  "working_directory": "/app"
}
```

#### **Health Check Route** (`/health`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "port": "8080",
  "host": "your-app.railway.app",
  "timestamp": 1705312200
}
```

### **Step 3: Deployment Options**

#### **Option A: Gunicorn (Recommended)**
```bash
# Use the main Dockerfile with Gunicorn
git add .
git commit -m "Add comprehensive debugging and test routes"
git push origin main
```

#### **Option B: Simplified Flask Server**
```bash
# If Gunicorn fails, try the simplified Dockerfile
cp Dockerfile.simple Dockerfile
git add .
git commit -m "Switch to simplified Dockerfile"
git push origin main
```

### **Step 4: Monitor Deployment**

#### **Expected Railway Logs**
```
=== RAILWAY DEBUG STARTUP ===
🔍 COMPREHENSIVE STARTUP DEBUG
============================================================
📅 Timestamp: 2024-01-15 10:30:00
🐍 Python version: 3.11.x
📂 Working directory: /app
🌍 Environment variables:
   PORT: 8080
   RAILWAY_ENVIRONMENT: production
   PYTHONPATH: NOT_SET
📁 Files in current directory:
   server.py: 15,000 bytes
   requirements.txt: 500 bytes
   blackhat2025_model.dill: 14,680,064 bytes
🤖 MODEL LOADING DEBUG
========================================
📍 BASE_DIR: /app
📄 Model path: /app/blackhat2025_model.dill
📊 Model file size: 14,680,064 bytes (14.0 MB)
✅ Dill model loaded successfully!
✅ SUCCESS: Model loaded successfully for production
🚀 AUTHENTICATION SERVER (PRODUCTION MODE)
============================================================
[INFO] Starting gunicorn 20.1.0
[INFO] Listening at: http://0.0.0.0:8080
```

#### **Success Indicators**
- ✅ Environment variables are set correctly
- ✅ Model file exists and loads successfully
- ✅ All Python imports work
- ✅ Gunicorn starts without errors
- ✅ Health check responds with 200 status

### **Step 5: Test Endpoints**

#### **Test Flask App**
```bash
curl https://your-app.railway.app/test
```

#### **Check Debug Info**
```bash
curl https://your-app.railway.app/debug
```

#### **Verify Health**
```bash
curl https://your-app.railway.app/health
```

#### **Test Main App**
```bash
curl https://your-app.railway.app/
```

### **Step 6: Troubleshooting**

#### **If Model Loading Fails**
1. Check if `blackhat2025_model.dill` is in the repository
2. Verify file size (should be ~14MB)
3. Check Railway memory limits
4. Try simplified Dockerfile

#### **If Gunicorn Fails**
1. Check `/debug` endpoint for system info
2. Verify all imports work
3. Try Flask dev server instead
4. Check Railway logs for specific errors

#### **If Port Issues**
1. Verify PORT environment variable
2. Check Railway port configuration
3. Test with different port binding

#### **If Health Check Fails**
1. Check if server is actually running
2. Verify health check path in Railway settings
3. Check for timeout issues

### **Step 7: Common Error Patterns**

#### **Missing Model File**
```
❌ Model file not found: /app/blackhat2025_model.dill
🔍 Looking for similar files:
   Found: other_model.dill
```
**Fix**: Ensure model file is in repository

#### **Import Failures**
```
❌ torch - FAILED - No module named 'torch'
❌ cv2 - FAILED - No module named 'cv2'
```
**Fix**: Check requirements.txt and system dependencies

#### **Memory Issues**
```
❌ Model loading failed: [Errno 12] Cannot allocate memory
```
**Fix**: Increase Railway memory allocation

#### **Port Binding Issues**
```
❌ Port binding test failed: Address already in use
```
**Fix**: Check Railway port configuration

### **Step 8: Final Verification**

#### **All Tests Should Pass**
- [ ] `/test` returns Flask working status
- [ ] `/debug` shows model loaded and files present
- [ ] `/health` returns healthy status
- [ ] `/` serves the main app
- [ ] `/api/predict` accepts POST requests

#### **Railway Dashboard**
- [ ] Deployment status is "Deployed"
- [ ] Health checks are passing
- [ ] No error logs
- [ ] Resource usage is within limits

---

**🎯 Goal**: Use this checklist to systematically verify each component and identify the exact issue preventing successful deployment. 