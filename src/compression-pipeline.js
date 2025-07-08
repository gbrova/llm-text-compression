import { LLMRanker } from './llm-ranker.js';
import { HuffmanRankCompressor } from './huffman-compression.js';
import { RankerCache } from './ranker-cache.js';
import { promisify } from 'util';
import zlib from 'zlib';

// Try to import sqlite3, fall back to mock if not available
let Database;
try {
    const sqlite3 = await import('sqlite3');
    Database = sqlite3.Database;
} catch (error) {
    // Mock Database class for testing without sqlite3
    Database = class MockDatabase {
        constructor() {}
        run() {}
        get(query, params, callback) { callback(null, null); }
        all(query, params, callback) { callback(null, []); }
        close() {}
    };
}

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

/**
 * Results from a compression experiment
 */
export class CompressionResult {
    constructor(options = {}) {
        this.method = options.method || '';
        this.originalSize = options.originalSize || 0;
        this.compressedSize = options.compressedSize || 0;
        this.compressionRatio = options.compressionRatio || 0;
        this.compressionTime = options.compressionTime || 0;
        this.decompressionTime = options.decompressionTime || 0;
        this.modelName = options.modelName || null;
        this.contextLength = options.contextLength || null;
        this.textSample = options.textSample || '';
    }

    /**
     * Get compression as percentage reduction
     * @returns {number} Compression percentage
     */
    get compressionPercentage() {
        return (1 - this.compressionRatio) * 100;
    }
}

/**
 * Main compression pipeline with database storage for results
 */
export class CompressionPipeline {
    /**
     * Initialize the compression pipeline
     * @param {Object} options - Configuration options
     * @param {string} options.dbPath - Path to SQLite database
     * @param {boolean} options.enableCache - Whether to enable caching
     */
    constructor(options = {}) {
        this.dbPath = options.dbPath || 'compression_results.db';
        this.enableCache = options.enableCache !== false;
        
        this.ranker = null;
        this.cache = this.enableCache ? new RankerCache(this.dbPath) : null;
        this.huffmanCompressor = new HuffmanRankCompressor();
        
        this._setupDatabase();
    }

    /**
     * Initialize the SQLite database
     * @private
     */
    _setupDatabase() {
        const db = new Database(this.dbPath);
        
        // Create compression results table
        db.run(`
            CREATE TABLE IF NOT EXISTS compression_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                method TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                compression_time REAL NOT NULL,
                decompression_time REAL NOT NULL,
                model_name TEXT,
                context_length INTEGER,
                text_sample TEXT,
                metadata TEXT
            )
        `);
        
        db.close();
    }

    /**
     * Get or create LLM ranker instance
     * @param {string} modelName - Model name
     * @param {number} maxContextLength - Maximum context length
     * @param {number} batchSize - Batch size
     * @returns {LLMRanker} Ranker instance
     * @private
     */
    _getRanker(modelName = 'gpt2', maxContextLength = null, batchSize = 1) {
        if (!this.ranker || 
            this.ranker.modelName !== modelName ||
            this.ranker.maxContextLength !== maxContextLength ||
            this.ranker.batchSize !== batchSize) {
            
            this.ranker = new LLMRanker({
                modelName,
                maxContextLength,
                batchSize
            });
        }
        
        return this.ranker;
    }

    /**
     * Get token ranks with optional caching
     * @param {string} text - Input text
     * @param {LLMRanker} ranker - Ranker instance
     * @returns {Array<number>} Token ranks
     */
    async getTokenRanksCached(text, ranker) {
        if (this.enableCache && this.cache) {
            return await this.cache.getTokenRanksCached(text, ranker);
        } else {
            return await ranker.getTokenRanks(text);
        }
    }

    /**
     * Get string from token ranks with optional caching
     * @param {Array<number>} ranks - Token ranks
     * @param {LLMRanker} ranker - Ranker instance
     * @returns {string} Generated text
     */
    async getStringFromTokenRanksCached(ranks, ranker) {
        if (this.enableCache && this.cache) {
            return await this.cache.getStringFromTokenRanksCached(ranks, ranker);
        } else {
            return await ranker.getStringFromTokenRanks(ranks);
        }
    }

    /**
     * Create a compression result object
     * @param {string} originalText - Original text
     * @param {Buffer} compressedData - Compressed data
     * @param {number} compressionTime - Compression time
     * @param {number} decompressionTime - Decompression time
     * @param {string} methodName - Method name
     * @param {string} modelName - Model name
     * @param {number} contextLength - Context length
     * @returns {CompressionResult} Result object
     * @private
     */
    _createCompressionResult(originalText, compressedData, compressionTime, 
                           decompressionTime, methodName, modelName = null, 
                           contextLength = null) {
        const originalSize = Buffer.byteLength(originalText, 'utf8');
        const compressedSize = compressedData.length;
        
        return new CompressionResult({
            method: methodName,
            originalSize,
            compressedSize,
            compressionRatio: compressedSize / originalSize,
            compressionTime,
            decompressionTime,
            modelName,
            contextLength,
            textSample: originalText.substring(0, 100)
        });
    }

    /**
     * Compress text using LLM rank encoding + gzip
     * @param {string} text - Input text
     * @param {string} modelName - Model name
     * @param {number} maxContextLength - Maximum context length
     * @returns {Object} {compressedData: Buffer, result: CompressionResult}
     */
    async compressWithLLMRanks(text, modelName = 'gpt2', maxContextLength = null) {
        const ranker = this._getRanker(modelName, maxContextLength);
        
        // Phase 1: Get token ranks
        const startTime = Date.now();
        const ranks = await this.getTokenRanksCached(text, ranker);
        
        // Phase 2: Compress ranks with gzip
        const ranksBuffer = Buffer.from(JSON.stringify(ranks));
        const compressedData = await gzip(ranksBuffer);
        const compressionTime = (Date.now() - startTime) / 1000;
        
        // Test decompression
        const decompStartTime = Date.now();
        const decompressedRanks = JSON.parse((await gunzip(compressedData)).toString());
        const reconstructedText = await this.getStringFromTokenRanksCached(decompressedRanks, ranker);
        const decompressionTime = (Date.now() - decompStartTime) / 1000;
        
        // Verify round-trip accuracy
        if (reconstructedText !== text) {
            throw new Error('Round-trip compression failed - reconstructed text doesn\'t match original');
        }
        
        const result = this._createCompressionResult(
            text, compressedData, compressionTime, decompressionTime,
            'llm_ranks_gzip', modelName, maxContextLength
        );
        
        return { compressedData, result };
    }

    /**
     * Compress text using LLM rank encoding + Huffman coding
     * @param {string} text - Input text
     * @param {string} modelName - Model name
     * @param {number} maxContextLength - Maximum context length
     * @returns {Object} {compressedData: Buffer, result: CompressionResult}
     */
    async compressWithLLMRanksHuffman(text, modelName = 'gpt2', maxContextLength = null) {
        const ranker = this._getRanker(modelName, maxContextLength);
        
        // Phase 1: Get token ranks
        const ranks = await this.getTokenRanksCached(text, ranker);
        
        // Phase 2: Compress ranks with Huffman coding
        const { compressedData, compressionTime } = 
            this.huffmanCompressor.compressRanksWithTiming(ranks, 'basic');
        
        // Phase 3: Test decompression
        const { decompressedRanks, decompressionTime } = 
            this.huffmanCompressor.decompressRanksWithTiming(compressedData, 'basic');
        const reconstructedText = await this.getStringFromTokenRanksCached(decompressedRanks, ranker);
        
        // Verify round-trip accuracy
        if (reconstructedText !== text) {
            throw new Error('Round-trip compression failed - reconstructed text doesn\'t match original');
        }
        
        const result = this._createCompressionResult(
            text, compressedData, compressionTime, decompressionTime,
            'llm_ranks_huffman', modelName, maxContextLength
        );
        
        return { compressedData, result };
    }

    /**
     * Compress text using LLM rank encoding + Huffman coding with Zipf distribution
     * @param {string} text - Input text
     * @param {string} modelName - Model name
     * @param {number} maxContextLength - Maximum context length
     * @returns {Object} {compressedData: Buffer, result: CompressionResult}
     */
    async compressWithLLMRanksHuffmanZipf(text, modelName = 'gpt2', maxContextLength = null) {
        const ranker = this._getRanker(modelName, maxContextLength);
        
        // Phase 1: Get token ranks
        const ranks = await this.getTokenRanksCached(text, ranker);
        
        // Phase 2: Compress ranks with Zipf-based Huffman coding
        const { compressedData, compressionTime } = 
            this.huffmanCompressor.compressRanksWithTiming(ranks, 'zipf');
        
        // Phase 3: Test decompression
        const { decompressedRanks, decompressionTime } = 
            this.huffmanCompressor.decompressRanksWithTiming(compressedData, 'zipf');
        const reconstructedText = await this.getStringFromTokenRanksCached(decompressedRanks, ranker);
        
        // Verify round-trip accuracy
        if (reconstructedText !== text) {
            throw new Error('Round-trip compression failed - reconstructed text doesn\'t match original');
        }
        
        const result = this._createCompressionResult(
            text, compressedData, compressionTime, decompressionTime,
            'llm_ranks_huffman_zipf', modelName, maxContextLength
        );
        
        return { compressedData, result };
    }

    /**
     * Compress text using LLM rank encoding + Huffman coding with Zipf distribution (binary format)
     * @param {string} text - Input text
     * @param {string} modelName - Model name
     * @param {number} maxContextLength - Maximum context length
     * @returns {Object} {compressedData: Buffer, result: CompressionResult}
     */
    async compressWithLLMRanksHuffmanZipfBytes(text, modelName = 'gpt2', maxContextLength = null) {
        const ranker = this._getRanker(modelName, maxContextLength);
        
        // Phase 1: Get token ranks
        const ranks = await this.getTokenRanksCached(text, ranker);
        
        // Phase 2: Compress ranks with Zipf-based Huffman coding (binary format)
        const { compressedData, compressionTime } = 
            this.huffmanCompressor.compressRanksWithTiming(ranks, 'zipf_bytes');
        
        // Phase 3: Test decompression
        const { decompressedRanks, decompressionTime } = 
            this.huffmanCompressor.decompressRanksWithTiming(compressedData, 'zipf_bytes');
        const reconstructedText = await this.getStringFromTokenRanksCached(decompressedRanks, ranker);
        
        // Verify round-trip accuracy
        if (reconstructedText !== text) {
            throw new Error('Round-trip compression failed - reconstructed text doesn\'t match original');
        }
        
        const result = this._createCompressionResult(
            text, compressedData, compressionTime, decompressionTime,
            'llm_ranks_huffman_zipf_bytes', modelName, maxContextLength
        );
        
        return { compressedData, result };
    }

    /**
     * Baseline: compress raw text with gzip
     * @param {string} text - Input text
     * @returns {Object} {compressedData: Buffer, result: CompressionResult}
     */
    async compressWithRawGzip(text) {
        const textBuffer = Buffer.from(text, 'utf8');
        
        // Compression
        const startTime = Date.now();
        const compressedData = await gzip(textBuffer);
        const compressionTime = (Date.now() - startTime) / 1000;
        
        // Test decompression
        const decompStartTime = Date.now();
        const decompressedData = await gunzip(compressedData);
        const decompressionTime = (Date.now() - decompStartTime) / 1000;
        
        // Verify round-trip
        if (decompressedData.toString('utf8') !== text) {
            throw new Error('Round-trip compression failed');
        }
        
        const result = this._createCompressionResult(
            text, compressedData, compressionTime, decompressionTime,
            'raw_gzip'
        );
        
        return { compressedData, result };
    }

    /**
     * Baseline: tokenize then compress token IDs with gzip
     * @param {string} text - Input text
     * @param {string} modelName - Model name
     * @returns {Object} {compressedData: Buffer, result: CompressionResult}
     */
    async compressWithTokenizerGzip(text, modelName = 'gpt2') {
        const ranker = this._getRanker(modelName);
        
        // Tokenize
        const startTime = Date.now();
        const tokenIds = await ranker.tokenize(text);
        const tokenBuffer = Buffer.from(JSON.stringify(tokenIds));
        const compressedData = await gzip(tokenBuffer);
        const compressionTime = (Date.now() - startTime) / 1000;
        
        // Test decompression
        const decompStartTime = Date.now();
        const decompressedTokens = JSON.parse((await gunzip(compressedData)).toString());
        const reconstructedText = await ranker.decode(decompressedTokens);
        const decompressionTime = (Date.now() - decompStartTime) / 1000;
        
        const result = this._createCompressionResult(
            text, compressedData, compressionTime, decompressionTime,
            'tokenizer_gzip', modelName
        );
        
        return { compressedData, result };
    }

    /**
     * Run all compression methods on text and return results
     * @param {string} text - Input text
     * @param {string} modelName - Model name
     * @param {number} maxContextLength - Maximum context length
     * @param {boolean} saveToDb - Whether to save results to database
     * @param {number} batchSize - Batch size for processing
     * @returns {Object} Dictionary mapping method names to results
     */
    async runCompressionBenchmark(text, modelName = 'gpt2', maxContextLength = null, 
                                  saveToDb = true, batchSize = 4) {
        const results = {};
        
        // Define compression methods to test
        const methods = [
            {
                name: 'llm_ranks_gzip',
                func: () => this.compressWithLLMRanks(text, modelName, maxContextLength)
            },
            {
                name: 'llm_ranks_huffman',
                func: () => this.compressWithLLMRanksHuffman(text, modelName, maxContextLength)
            },
            {
                name: 'llm_ranks_huffman_zipf',
                func: () => this.compressWithLLMRanksHuffmanZipf(text, modelName, maxContextLength)
            },
            {
                name: 'llm_ranks_huffman_zipf_bytes',
                func: () => this.compressWithLLMRanksHuffmanZipfBytes(text, modelName, maxContextLength)
            },
            {
                name: 'raw_gzip',
                func: () => this.compressWithRawGzip(text)
            },
            {
                name: 'tokenizer_gzip',
                func: () => this.compressWithTokenizerGzip(text, modelName)
            }
        ];
        
        // Run each method
        for (const method of methods) {
            try {
                const { result } = await method.func();
                results[method.name] = result;
                
                if (saveToDb) {
                    await this._saveResultToDb(result);
                }
            } catch (error) {
                console.error(`${method.name} compression failed:`, error.message);
                results[method.name] = null;
            }
        }
        
        return results;
    }

    /**
     * Save a compression result to the database
     * @param {CompressionResult} result - Result to save
     * @private
     */
    async _saveResultToDb(result) {
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            const query = `
                INSERT INTO compression_results (
                    timestamp, method, original_size, compressed_size, compression_ratio,
                    compression_time, decompression_time, model_name, context_length, 
                    text_sample, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            `;
            
            const values = [
                new Date().toISOString(),
                result.method,
                result.originalSize,
                result.compressedSize,
                result.compressionRatio,
                result.compressionTime,
                result.decompressionTime,
                result.modelName,
                result.contextLength,
                result.textSample,
                JSON.stringify(result)
            ];
            
            db.run(query, values, function(err) {
                db.close();
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }

    /**
     * Retrieve compression results from database
     * @param {string} method - Filter by method (optional)
     * @returns {Array} List of results
     */
    async getResultsFromDb(method = null) {
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            let query = 'SELECT * FROM compression_results ORDER BY timestamp DESC';
            let params = [];
            
            if (method) {
                query = 'SELECT * FROM compression_results WHERE method = ? ORDER BY timestamp DESC';
                params = [method];
            }
            
            db.all(query, params, (err, rows) => {
                db.close();
                if (err) {
                    reject(err);
                } else {
                    resolve(rows);
                }
            });
        });
    }

    /**
     * Print a formatted comparison table of compression results
     * @param {Object} results - Results dictionary
     */
    printComparisonTable(results) {
        const successfulResults = Object.entries(results).filter(([, result]) => result !== null);
        
        if (successfulResults.length === 0) {
            console.log('No successful compression results to display.');
            return;
        }
        
        console.log('\n' + '='.repeat(100));
        console.log('COMPRESSION COMPARISON RESULTS');
        console.log('='.repeat(100));
        
        // Table header
        console.log(
            'Method'.padEnd(20) + 
            'Original'.padEnd(10) + 
            'Compressed'.padEnd(10) + 
            'Ratio'.padEnd(8) + 
            'Reduction'.padEnd(10) + 
            'Comp (s)'.padEnd(10) + 
            'Decomp (s)'.padEnd(10)
        );
        console.log('-'.repeat(100));
        
        // Sort by compression ratio (best first)
        const sortedResults = successfulResults.sort((a, b) => a[1].compressionRatio - b[1].compressionRatio);
        
        for (const [method, result] of sortedResults) {
            console.log(
                method.padEnd(20) + 
                result.originalSize.toString().padEnd(10) + 
                result.compressedSize.toString().padEnd(10) + 
                result.compressionRatio.toFixed(3).padEnd(8) + 
                `${result.compressionPercentage.toFixed(1)}%`.padEnd(10) + 
                result.compressionTime.toFixed(3).padEnd(10) + 
                result.decompressionTime.toFixed(3).padEnd(10)
            );
        }
        
        console.log('-'.repeat(100));
        
        // Show best performing method
        if (sortedResults.length > 0) {
            const [bestMethod, bestResult] = sortedResults[0];
            console.log(`Best compression: ${bestMethod} (${bestResult.compressionPercentage.toFixed(1)}% reduction)`);
        }
        
        console.log('='.repeat(100));
    }
}

/**
 * Load a text file from the datasets directory
 * @param {string} filename - Filename to load
 * @returns {string} File contents
 */
export async function loadDatasetFile(filename) {
    const fs = await import('fs');
    const path = await import('path');
    
    const filePath = path.join('datasets', filename);
    
    if (!fs.existsSync(filePath)) {
        throw new Error(`Dataset file not found: ${filePath}`);
    }
    
    return fs.readFileSync(filePath, 'utf8');
}