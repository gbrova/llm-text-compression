# Python Implementation - LLM Compression

This is the original Python implementation of the LLM compression pipeline, featuring comprehensive research tools and a Flask-based web interface.

## Features

- **LLM-based token ranking** using HuggingFace transformers (GPT-2 and other models)
- **Multiple compression methods**:
  - Raw zlib compression (baseline)
  - Tokenizer + zlib compression
  - LLM ranks + zlib compression
  - Custom Huffman coding implementations
- **Interactive web UI** for token rank visualization
- **GPU acceleration** support
- **Comprehensive caching** system
- **Round-trip verification** to ensure perfect reconstruction

## Installation

### Prerequisites

- Python 3.8 or higher
- UV package manager (recommended) or pip

### Setup

1. **Clone the repository and navigate to the Python implementation:**
   ```bash
   cd python/
   ```

2. **Install dependencies using UV:**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive Web UI

🎨 **Interactive web interface** to visualize how the LLM ranks tokens:

```bash
# Start the web UI (default port 8080)
uv run python rank_visualizer.py

# Or use a custom port
uv run python rank_visualizer.py --port 3000

# Or use the convenience script (auto-opens browser)
uv run python start_ui.py

# Custom port with convenience script
uv run python start_ui.py --port 3000

# Set port via environment variable
FLASK_PORT=3000 uv run python rank_visualizer.py
```

Then open http://localhost:8080 (or your custom port) in your browser to:
- Enter text and see each token highlighted by its rank
- View rank numbers below each token
- See color gradients from green (predictable) to red (surprising)
- Try example texts to understand how context affects ranking

### Command Line Usage

```python
from llm_ranker import LLMRanker

# Create ranker instance
ranker = LLMRanker(model_name="gpt2")

# Encode text as ranks
text = "Hello world"
ranks = ranker.get_token_ranks(text)
print(f"Ranks: {ranks}")

# Reconstruct original text
reconstructed = ranker.get_string_from_token_ranks(ranks)
print(f"Reconstructed: {reconstructed}")
```

### Running Compression Experiments

```bash
# Run the main compression pipeline
uv run python main.py

# Run specific compression tests
uv run python compression_pipeline.py
```

## Project Structure

```
python/
├── README.md                 # This file
├── pyproject.toml           # Project configuration
├── uv.lock                  # Dependency lock file
├── .python-version          # Python version specification
├── llm_ranker.py            # Core LLM ranking implementation
├── compression_pipeline.py  # Compression pipeline
├── huffman_compression.py   # Huffman coding implementation
├── ranker_cache.py          # Caching system
├── rank_visualizer.py       # Web UI for token rank visualization
├── start_ui.py              # Convenience script to start the web UI
├── main.py                  # Entry point for experiments
├── templates/               # HTML templates for web UI
│   └── index.html
├── test_ranker.py           # Core ranking tests
├── test_flask_app.py        # Web UI tests
└── test_rank_visualizer.py  # Visualization tests
```

## Key Components

### LLMRanker
Core class for encoding text as token ranks and reconstructing text from ranks:
- GPT-2 integration using HuggingFace transformers
- Configurable context length
- GPU acceleration support
- Round-trip verification
- Comprehensive rank validation

### Compression Pipeline
Multiple compression strategies for rank sequences:
- Baseline comparisons (raw zlib, tokenizer+zlib)
- LLM ranks + traditional compression
- Custom Huffman coding implementations
- Performance benchmarking

### Web Visualization
Flask-based web interface:
- Real-time token rank visualization
- Interactive text input
- Color-coded token display
- Example text library
- Mobile-friendly responsive design

### Caching System
Efficient caching for expensive operations:
- Token rank caching
- String reconstruction caching
- Configurable cache backends
- Performance optimization

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest test_ranker.py -v
uv run pytest test_flask_app.py -v
uv run pytest test_rank_visualizer.py -v
```

## Performance Optimization

### GPU Acceleration
Enable GPU support for faster processing:
```python
ranker = LLMRanker(model_name="gpt2", device="cuda")
```

### Context Length Tuning
Adjust context length for speed vs. accuracy:
```python
ranker = LLMRanker(model_name="gpt2", max_context_length=512)
```

### Batch Processing
Process multiple texts efficiently:
```python
ranker = LLMRanker(model_name="gpt2", batch_size=8)
```

## Experimental Features

### Model Selection
Support for different model architectures:
- GPT-2 (default)
- GPT-3.5 (via API)
- Other HuggingFace models

### Advanced Compression
- Parametric distribution modeling
- Adaptive context lengths
- Hierarchical compression strategies

## Troubleshooting

**Model loading issues:**
- Ensure sufficient disk space for model downloads
- Check internet connection for initial model download
- Verify CUDA installation for GPU acceleration

**Memory issues:**
- Reduce context length for large texts
- Use smaller batch sizes
- Enable gradient checkpointing for large models

**Web UI not starting:**
- Check port availability
- Verify Flask dependencies are installed
- Try different port numbers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.