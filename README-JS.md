# LLM Text Compression - JavaScript Port

This is a JavaScript/Node.js port of the Python LLM text compression pipeline. It provides LLM-based text compression using transformers.js for inference, with multiple compression algorithms and comprehensive benchmarking capabilities.

## Features

- **LLM-based token ranking** using transformers.js (GPT-2 and other models)
- **Multiple compression methods**:
  - Raw gzip compression (baseline)
  - Tokenizer + gzip compression  
  - LLM ranks + gzip compression
  - LLM ranks + Huffman coding
  - LLM ranks + Huffman with Zipf distribution
- **Batched processing** for improved performance
- **Comprehensive round-trip testing** to ensure data integrity
- **SQLite-based result caching** and storage
- **Detailed benchmarking** and performance analysis

## Installation

### Prerequisites

- Node.js 18.0.0 or higher
- NPM or Yarn package manager

### Setup

1. **Install core dependencies:**
   ```bash
   npm install
   ```

2. **Install optional dependencies for full functionality:**
   ```bash
   npm run install-deps
   ```

3. **For Huffman coding (if available):**
   ```bash
   npm install huffman-coding
   ```

## Usage

### Quick Start

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

### Huffman Compression

```javascript
import { HuffmanRankCompressor } from './src/huffman-compression.js';

const compressor = new HuffmanRankCompressor();
const ranks = [1, 2, 1, 3, 2, 1, 4, 1, 2];

// Basic Huffman compression
const compressed = compressor.compressBasic(ranks);
const decompressed = compressor.decompressBasic(compressed);

// Zipf-based Huffman compression
const zipfCompressed = compressor.compressZipf(ranks);
const zipfDecompressed = compressor.decompressZipf(zipfCompressed);

// With timing information
const { compressedData, compressionTime } = 
    compressor.compressRanksWithTiming(ranks, 'zipf_bytes');
```

## Scripts

### Running the Demo
```bash
npm start
```

### Running Benchmarks
```bash
npm run benchmark
```

### Running Tests
```bash
npm test
```

## Project Structure

```
src/
├── compression-pipeline.js    # Main compression pipeline
├── llm-ranker.js             # LLM-based token ranking
├── huffman-compression.js    # Huffman coding utilities
├── ranker-cache.js          # SQLite-based caching
├── benchmark.js             # Benchmarking script
├── index.js                 # Main entry point
└── test/                    # Test suite
    ├── compression-pipeline.test.js
    ├── huffman-compression.test.js
    └── llm-ranker.test.js
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

## Performance Features

### Caching
- **Token rank caching**: Expensive LLM computations are cached in SQLite
- **String generation caching**: Reconstruction results are cached
- **Configurable**: Can be enabled/disabled per pipeline instance

### Batching
- **Parallel processing**: Process multiple token chunks simultaneously
- **Configurable batch size**: Adjust based on available memory and compute
- **Automatic chunking**: Splits long sequences into optimal chunks

### Database Storage
- **SQLite backend**: All results stored in local database
- **Comprehensive metadata**: Timing, model info, compression ratios
- **Querying support**: Filter and analyze results by method

## Testing

The test suite is designed to work even when optional dependencies are missing:

```bash
npm test
```

Tests cover:
- **Round-trip verification**: Ensures perfect reconstruction
- **Edge cases**: Empty inputs, large values, special characters
- **Performance**: Timing and memory usage
- **Batch processing**: Chunking logic and parallel execution
- **Database operations**: Storage and retrieval

## Limitations & Notes

### Model Dependencies
- **transformers.js**: May require significant download for model weights
- **Local execution**: Models run locally (no API calls)
- **Memory requirements**: Large models need sufficient RAM

### Optional Dependencies
- **huffman-coding**: May need separate installation or custom implementation
- **Graceful degradation**: Tests and demos work without optional deps
- **Platform compatibility**: Some native dependencies may require compilation

### Performance Considerations
- **First run**: Model loading takes time on initial execution
- **Memory usage**: Keep context length reasonable for available RAM
- **Disk space**: Model weights and cache database can be large

## Development

### Adding New Compression Methods

1. Implement method in `CompressionPipeline` class
2. Add to `runCompressionBenchmark` method list
3. Create corresponding tests
4. Update documentation

### Extending LLM Support

1. Modify `LLMRanker` constructor for new model types
2. Update tokenization and inference logic
3. Test with different model architectures
4. Document model-specific requirements

## Comparison with Python Version

### Similarities
- **Same algorithms**: All compression methods ported
- **Same test patterns**: Round-trip verification preserved  
- **Same database schema**: Compatible result storage
- **Same benchmarking**: Comparable performance analysis

### Differences
- **Runtime**: Node.js vs Python runtime differences
- **Dependencies**: Different library ecosystem
- **Model loading**: transformers.js vs transformers (Python)
- **Performance**: May vary due to runtime and library differences

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original Python implementation by the research team
- Hugging Face for transformers.js library
- Node.js community for ecosystem support