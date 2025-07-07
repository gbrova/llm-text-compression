#!/usr/bin/env python3
"""
Convenience script to start the LLM Rank Visualizer web UI.
"""

import subprocess
import sys
import webbrowser
import time
import threading
import argparse
import os

def start_server(port=8080, host='127.0.0.1'):
    """Start the Flask server."""
    try:
        cmd = [sys.executable, "rank_visualizer.py", "--port", str(port), "--host", host]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def open_browser(port=8080, host='127.0.0.1'):
    """Open browser after a delay."""
    time.sleep(2)  # Wait for server to start
    url = f"http://{host}:{port}"
    try:
        webbrowser.open(url)
        print(f"🌐 Opened browser at {url}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please manually open {url} in your browser")

def main():
    """Main function with configurable port."""
    parser = argparse.ArgumentParser(description='Start LLM Rank Visualizer with auto-opening browser')
    parser.add_argument('--port', '-p', type=int, default=8080, 
                        help='Port to run the web server on (default: 8080)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind the web server to (default: 127.0.0.1)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Don\'t automatically open browser')
    
    args = parser.parse_args()
    
    # Allow port to be set via environment variable
    port = int(os.environ.get('FLASK_PORT', args.port))
    host = os.environ.get('FLASK_HOST', args.host)
    
    print("🚀 Starting LLM Rank Visualizer...")
    print("📊 This will load the GPT-2 model and start a web server")
    print(f"🌐 Server will be available at http://{host}:{port}")
    print("⏳ Please wait while the model loads...")
    print()
    
    # Start browser in background (unless disabled)
    if not args.no_browser:
        browser_thread = threading.Thread(target=lambda: open_browser(port, host), daemon=True)
        browser_thread.start()
    
    # Start server (this will block)
    start_server(port, host)

if __name__ == "__main__":
    main()