# LLM Rank Visualizer - JavaScript Client

This is a standalone JavaScript client for visualizing LLM token rankings, built using transformers.js. It runs entirely in the browser and uses the existing `src/llm-ranker.js` compression implementation.

## Features

- **Client-side processing**: Uses transformers.js to load and run GPT-2 directly in the browser
- **Token ranking visualization**: Shows how each token is ranked by the language model given the preceding context
- **Color-coded display**: Tokens are colored from green (low rank/predictable) to red (high rank/unpredictable)
- **Interactive interface**: Click on example texts or enter your own text to analyze
- **Responsive design**: Works on desktop and mobile devices

## Files

- `visualizer-client.html` - Main visualizer client with proper dependency handling
- `server.js` - Node.js HTTP server with module resolution
- `README-VISUALIZER.md` - This documentation

## Usage

### Quick Start

```bash
npm run serve
```

This will:
- Start a Node.js HTTP server on port 8000
- Automatically open your browser to the visualizer
- Serve the files with proper CORS headers and module resolution

### Alternative Commands

```bash
npm run dev      # Same as npm run serve
npm start        # Runs the main compression pipeline
```

## How It Works

1. **Model Loading**: The client loads the GPT-2 model using transformers.js from Hugging Face
2. **Tokenization**: Input text is tokenized into individual tokens
3. **Ranking**: For each token position, the model predicts what tokens are most likely given the preceding context
4. **Visualization**: Each token is colored based on its rank in the model's predictions

## Model Information

- **Model**: GPT-2 (DistilGPT-2 via transformers.js)
- **Size**: Approximately 500MB download on first use
- **Caching**: The model is cached by the browser for subsequent uses
- **Performance**: Processing speed depends on text length and device capabilities

## Color Legend

- **Green**: Low rank (1-33% of max rank) - highly predictable tokens
- **Yellow**: Medium rank (34-66% of max rank) - moderately predictable tokens  
- **Red**: High rank (67-100% of max rank) - less predictable tokens

## Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support (may require HTTPS for some features)
- **Mobile**: Supported but may be slower due to model size

## Performance Notes

- Initial model loading takes 1-2 minutes on first use
- Text analysis speed depends on text length and device
- For best performance, use shorter texts (< 50 tokens)
- The model runs entirely in your browser - no data is sent to external servers

## Troubleshooting

**Model won't load:**
- Check your internet connection
- Try refreshing the page
- Ensure you have enough free memory (model requires ~1GB RAM)

**Slow performance:**
- Use shorter input texts
- Close other browser tabs to free memory
- Try on a device with more RAM

**CORS or module errors:**
- Always use `npm run serve` - don't open HTML files directly
- Make sure you've run `npm install` first

## Technical Details

The client uses:
- **transformers.js**: Hugging Face's JavaScript library for running transformers models
- **GPT-2**: OpenAI's language model for token ranking
- **Vanilla JavaScript**: No external frameworks required
- **ES6 modules**: Modern JavaScript with import/export syntax
- **CDN imports**: transformers.js loaded from Hugging Face CDN

## Comparison with Python Version

This JavaScript client provides the same core functionality as the Python `rank_visualizer.py` but with these differences:

**Advantages:**
- No backend server required
- Runs entirely in browser
- No Python dependencies
- Easy to share and deploy
- Same ranking algorithm as the compression pipeline

**Limitations:**
- Larger initial download (model needs to be downloaded)
- May be slower than Python version
- Limited to smaller models that can run in browser
- Requires modern browser with WebAssembly support