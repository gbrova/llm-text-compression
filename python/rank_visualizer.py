#!/usr/bin/env python3
"""
Web UI for visualizing LLM token rank encoding.
Provides an interface to input text and see tokens highlighted by their rank values.
"""

import json
import os
import argparse
from flask import Flask, render_template, request, jsonify
from llm_ranker import LLMRanker
from compression_pipeline import CompressionPipeline
from typing import List, Dict, Tuple
import colorsys

app = Flask(__name__)

# Global ranker instance (initialized on first use)
ranker = None
compression_pipeline = None

def get_ranker():
    """Get or initialize the global ranker instance."""
    global ranker
    if ranker is None:
        ranker = LLMRanker(model_name="gpt2")
    return ranker

def get_compression_pipeline():
    """Get or initialize the global compression pipeline instance."""
    global compression_pipeline
    if compression_pipeline is None:
        compression_pipeline = CompressionPipeline(enable_cache=True)
    return compression_pipeline

def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert RGB values (0-1) to hex color string."""
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def rank_to_color(rank: int, max_rank: int) -> str:
    """Convert a rank to a color on a gradient from green (low rank) to red (high rank)."""
    if max_rank <= 1:
        return rgb_to_hex(0.2, 0.8, 0.2)  # Default green for single token
    
    # Normalize rank to 0-1 scale
    normalized = (rank - 1) / (max_rank - 1)
    
    # Use HSV color space for better gradient
    # Hue: 120° (green) to 0° (red)
    hue = (1 - normalized) * 120 / 360  # Convert to 0-1 scale
    saturation = 0.7
    value = 0.9
    
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return rgb_to_hex(r, g, b)

def get_tokens_and_ranks(text: str) -> List[Dict]:
    """
    Get tokens and their ranks for visualization.
    
    Returns:
        List of dictionaries with 'token', 'rank', and 'color' keys
    """
    if not text.strip():
        return []
    
    ranker_instance = get_ranker()
    
    # Get ranks
    ranks = ranker_instance.get_token_ranks(text)
    
    # Get tokens by encoding then decoding each token individually
    encoded = ranker_instance.tokenizer.encode(text)
    tokens = []
    
    for token_id in encoded:
        token_text = ranker_instance.tokenizer.decode([token_id])
        tokens.append(token_text)
    
    # Ensure we have the same number of tokens and ranks
    if len(tokens) != len(ranks):
        raise ValueError(f"Mismatch: {len(tokens)} tokens but {len(ranks)} ranks")
    
    # Calculate colors
    max_rank = max(ranks) if ranks else 1
    
    result = []
    for token, rank in zip(tokens, ranks):
        result.append({
            'token': token,
            'rank': rank,
            'color': rank_to_color(rank, max_rank)
        })
    
    return result

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text and return token ranks for visualization."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to analyze'})
        
        # Get tokens and ranks
        tokens_with_ranks = get_tokens_and_ranks(text)
        
        return jsonify({
            'success': True,
            'tokens': tokens_with_ranks,
            'total_tokens': len(tokens_with_ranks)
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

@app.route('/compress', methods=['POST'])
def compress_text():
    """Compress text using various methods and return results."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to compress'})
        
        # Get compression pipeline
        pipeline = get_compression_pipeline()
        
        # Run compression methods
        results = {}
        original_size = len(text.encode('utf-8'))
        results['original_size'] = original_size
        results['original_text'] = {'size': original_size}
        
        # Define compression methods to run
        compression_methods = [
            ('llm_ranks_zlib', lambda: pipeline.compress_with_llm_ranks(text)),
            ('llm_ranks_huffman_zipf_bytes', lambda: pipeline.compress_with_llm_ranks_huffman_zipf_bytes(text)),
            ('raw_zlib', lambda: pipeline.compress_with_raw_zlib(text)),
            ('tokenizer_zlib', lambda: pipeline.compress_with_tokenizer_zlib(text))
        ]
        
        for method_name, method_func in compression_methods:
            try:
                _, result = method_func()
                results[method_name] = {
                    'compressed_size': result.compressed_size,
                    'compression_ratio': result.compression_ratio,
                    'compression_time': result.compression_time
                }
            except Exception as e:
                print(f"Compression method {method_name} failed: {e}")
                # Continue with other methods
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Compression failed: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': 'gpt2'})

def main():
    """Main function to run the Flask application with configurable port."""
    parser = argparse.ArgumentParser(description='LLM Rank Visualizer Web UI')
    parser.add_argument('--port', '-p', type=int, default=8080, 
                        help='Port to run the web server on (default: 8080)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind the web server to (default: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Allow port to be set via environment variable
    port = int(os.environ.get('FLASK_PORT', args.port))
    host = os.environ.get('FLASK_HOST', args.host)
    
    print("🚀 Starting LLM Rank Visualizer...")
    print("📊 Loading GPT-2 model (this may take a moment)...")
    
    # Initialize the ranker to load the model
    get_ranker()
    
    print("✅ Model loaded successfully!")
    print("🔧 Initializing compression pipeline...")
    
    # Initialize the compression pipeline
    get_compression_pipeline()
    
    print("✅ Compression pipeline ready!")
    print(f"🌐 Starting web server at http://{host}:{port}")
    print(f"💡 You can also set port via environment variable: FLASK_PORT={port}")
    
    app.run(debug=args.debug, host=host, port=port)

if __name__ == '__main__':
    main()
