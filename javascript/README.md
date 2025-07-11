# JavaScript Implementation - LLM Compression

This is a JavaScript/Node.js implementation of the LLM compression pipeline, featuring browser-compatible execution using transformers.js and comprehensive benchmarking capabilities.

## Features

- **Browser-compatible LLM processing** using transformers.js
- **Multiple compression methods**:
  - Raw gzip compression (baseline)
  - Tokenizer + gzip compression  
  - LLM ranks + gzip compression
  - LLM ranks + Huffman coding
  - LLM ranks + Huffman with Zipf distribution
- **Standalone visualizer client** that runs entirely in the browser
- **Batched processing** for improved performance
- **SQLite-based caching** and result storage
- **Comprehensive benchmarking** and performance analysis
- **Round-trip verification** to ensure perfect reconstruction

## Installation

### Prerequisites

- Node.js 18.0.0 or higher
- NPM or Yarn package manager

### Setup

1. **Navigate to the JavaScript implementation:**
   ```bash
   cd javascript/
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

## Usage

### Quick Start - Compression Pipeline

```javascript
import { CompressionPipeline } from './src/compression-pipeline.js';

// Create compression pipeline
const pipeline = new CompressionPipeline();

// Compress text with different methods
const text = "Hello world! This is a test of the compression pipeline.";

// Raw gzip compression (baseline)
const { compressedData, result } = await pipeline.compressWithRawGzip(text);
console.log(`Compression ratio: ${result.compressionRatio.toFixed(3)}`);
console.log(`Space saved: ${result.compressionPercentage.toFixed(1)}%`);

// Run full benchmark comparison
const results = await pipeline.runCompressionBenchmark(text);
pipeline.printComparisonTable(results);
```

### LLM Token Ranking

```javascript
import { LLMRanker, rankTokens } from './src/llm-ranker.js';

// Create ranker instance
const ranker = new LLMRanker({
    modelName: 'gpt2',
    maxContextLength: 1024,
    batchSize: 1
});

// Get token ranks for text
const text = "The quick brown fox";
const ranks = await ranker.getTokenRanks(text);
console.log('Token ranks:', ranks);

// Reconstruct text from ranks
const reconstructed = await ranker.getStringFromTokenRanks(ranks);
console.log('Reconstructed:', reconstructed);

// Convenience function
const quickRanks = await rankTokens("Hello world", "gpt2");
```

### Browser Visualizer

Launch the standalone browser-based visualizer:

```bash
npm run serve
```

This will:
- Start a Node.js HTTP server on port 8000
- Automatically open your browser to the visualizer
- Load the GPT-2 model directly in your browser
- Provide interactive token rank visualization

## Scripts

### Available Commands

```bash
# Run the main compression demo
npm start

# Launch the browser visualizer
npm run serve
npm run dev        # Same as serve

# Run comprehensive benchmarks
npm run benchmark

# Run the complete test suite
npm test
```

## Project Structure

```
javascript/
├── README.md                    # This file
├── package.json                 # Project configuration
├── package-lock.json           # Dependency lock file
├── server.js                   # Development server
├── visualizer-client.html      # Standalone browser visualizer
├── test-runner.js              # Test execution
├── test-real-roundtrip.js      # Real-world testing
├── src/                        # Source code
│   ├── compression-pipeline.js # Main compression pipeline
│   ├── llm-ranker.js           # LLM-based token ranking
│   ├── huffman-compression.js  # Huffman coding utilities
│   ├── ranker-cache.js         # SQLite-based caching
│   ├── benchmark.js            # Benchmarking script
│   ├── index.js                # Main entry point
│   └── test/                   # Test suite
│       ├── compression-pipeline.test.js
│       ├── huffman-compression.test.js
│       └── llm-ranker.test.js
```

## Compression Methods

### 1. Raw Gzip (Baseline)
Simple gzip compression of the original text.

### 2. Tokenizer + Gzip
Tokenize text using the model's tokenizer, then compress token IDs with gzip.

### 3. LLM Ranks + Gzip
1. Use LLM to compute likelihood ranks for each token
2. Compress the rank sequence with gzip

### 4. LLM Ranks + Huffman
1. Compute token ranks using LLM
2. Use Huffman coding based on actual rank frequencies

### 5. LLM Ranks + Huffman + Zipf
1. Compute token ranks using LLM  
2. Fit a Zipf distribution to the rank frequencies
3. Use parametric Huffman coding based on the fitted distribution

## Browser Visualizer

The standalone visualizer client features:

- **Client-side processing**: No server required, runs entirely in browser
- **Model loading**: Downloads and caches GPT-2 model locally
- **Interactive interface**: Real-time token rank visualization
- **Color coding**: Green (predictable) to red (surprising) tokens
- **Example texts**: Pre-loaded examples to demonstrate functionality

### Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support (may require HTTPS)
- **Mobile**: Supported but may be slower

## Performance Features

### Caching
- **Token rank caching**: Expensive LLM computations cached in SQLite
- **String generation caching**: Reconstruction results cached
- **Model caching**: Browser caches downloaded models

### Batching
- **Parallel processing**: Multiple token chunks processed simultaneously
- **Configurable batch size**: Adjust for memory/compute constraints
- **Automatic chunking**: Optimal chunk size determination

### Database Storage
- **SQLite backend**: All results stored locally
- **Comprehensive metadata**: Timing, model info, compression ratios
- **Query support**: Filter and analyze results

## Testing

The test suite works with or without optional dependencies:

```bash
npm test
```

Tests cover:
- **Round-trip verification**: Perfect reconstruction guarantee
- **Edge cases**: Empty inputs, special characters, large values
- **Performance**: Timing and memory usage validation
- **Batch processing**: Chunking and parallel execution
- **Database operations**: Storage and retrieval functionality

## Limitations & Considerations

### Model Dependencies
- **Initial download**: GPT-2 model requires ~500MB download
- **Memory requirements**: Models need sufficient RAM (~1GB)
- **Loading time**: First execution includes model loading overhead

### Browser Constraints
- **Model size**: Limited to models that can run in browser
- **Memory limits**: Browser memory constraints apply
- **Processing speed**: May be slower than server-side processing

### Performance Notes
- **First run**: Model loading takes 1-2 minutes initially
- **Text length**: Shorter texts process faster
- **Device capability**: Performance depends on available resources

## Development

### Adding New Compression Methods

1. Implement method in `CompressionPipeline` class
2. Add to `runCompressionBenchmark` method list
3. Create corresponding tests
4. Update documentation

### Extending Model Support

1. Modify `LLMRanker` constructor for new model types
2. Update tokenization and inference logic
3. Test with different model architectures
4. Document model-specific requirements

## Comparison with Python Implementation

### Similarities
- **Same algorithms**: All compression methods ported
- **Same test patterns**: Round-trip verification preserved  
- **Compatible results**: Comparable performance metrics
- **Same database schema**: Compatible result storage

### Differences
- **Runtime environment**: Browser vs. server execution
- **Model loading**: transformers.js vs. transformers (Python)
- **Dependency management**: NPM vs. Python package ecosystem
- **Deployment**: Can run entirely client-side

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.