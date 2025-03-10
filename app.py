import os
import sys
from pathlib import Path
import socket

# Add the project root directory to Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Now we can import our modules
from code.IV_ui.ui_flask import app

def print_server_urls(port):
    """Print server URLs for easy access"""
    print("\n" + "="*50)
    print(f"🚀 Ozart server running at:")
    print(f"   http://localhost:{port}")
    print(f"   http://{socket.gethostbyname(socket.gethostname())}:{port}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5001))
    
    # Create necessary directories
    output_dirs = [
        "output/images",
        "output/descriptions",
        "output/music_analysis"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Print URLs before starting server
    print_server_urls(port)
    
    # Run Flask app with production server
    from waitress import serve
    print(f"Starting production server on port {port}...")
    serve(app, host="0.0.0.0", port=port)
