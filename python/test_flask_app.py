#!/usr/bin/env python3
"""
Test the Flask application endpoints.
"""

import requests
import json
import time

def test_flask_endpoints(port=8080):
    """Test that Flask endpoints work correctly."""
    base_url = f"http://localhost:{port}"
    
    print("🧪 Testing Flask application...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to server: {e}")
        print("   Make sure the server is running with: uv run python rank_visualizer.py --port 8080")
        return False
    
    # Test analyze endpoint
    try:
        test_data = {"text": "Hello world"}
        response = requests.post(
            f"{base_url}/analyze", 
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Analyze endpoint passed")
                print(f"   Analyzed {data['total_tokens']} tokens")
                # Show first token
                if data['tokens']:
                    first_token = data['tokens'][0]
                    print(f"   First token: '{first_token['token']}' (rank: {first_token['rank']})")
            else:
                print(f"❌ Analyze endpoint returned error: {data.get('error')}")
                return False
        else:
            print(f"❌ Analyze endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Analyze request failed: {e}")
        return False
    
    print("✅ All Flask tests passed!")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Flask application endpoints')
    parser.add_argument('--port', '-p', type=int, default=8080,
                        help='Port the server is running on (default: 8080)')
    args = parser.parse_args()
    
    success = test_flask_endpoints(args.port)
    if not success:
        print(f"\n💡 To start the server, run: uv run python rank_visualizer.py --port {args.port}")
        print("   Then try this test again in another terminal.")