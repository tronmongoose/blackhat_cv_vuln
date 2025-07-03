# blackhat2025

This is a simple web application that uses a camera to authenticate users based on their face and a coffee mug or tea cup. The AI model runs on a server, and the app communicates with it to verify the user's identity.

To set up the application, follow these steps:

1. **Install Dependencies**: 
Use uv to install the required Python packages. Setup a venv:
```bash
uv venv
```
Then activate the virtual environment:
```bash
source venv/bin/activate  # On macOS/Linux
```
Install the required packages using the following command:
```bash
uv sync 
```

2. **Run the Server**: Start the server by running `python server.py`. This will load the AI model and start listening for requests.
3. **Open the App**: Open your web browser and navigate to `http://localhost:8080` to access the application.