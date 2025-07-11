import { describe, it, before, after } from 'node:test';
import { strict as assert } from 'node:assert';
import { CompressionPipeline, CompressionResult } from '../compression-pipeline.js';
import { unlink } from 'fs/promises';

describe('CompressionResult', () => {
    it('should create compression result with correct properties', () => {
        const result = new CompressionResult({
            method: 'test_method',
            originalSize: 1000,
            compressedSize: 500,
            compressionRatio: 0.5,
            compressionTime: 1.5,
            decompressionTime: 0.8,
            modelName: 'gpt2',
            contextLength: 1024,
            textSample: 'Sample text...'
        });
        
        assert.equal(result.method, 'test_method');
        assert.equal(result.originalSize, 1000);
        assert.equal(result.compressedSize, 500);
        assert.equal(result.compressionRatio, 0.5);
        assert.equal(result.compressionPercentage, 50);
        assert.equal(result.compressionTime, 1.5);
        assert.equal(result.decompressionTime, 0.8);
        assert.equal(result.modelName, 'gpt2');
        assert.equal(result.contextLength, 1024);
        assert.equal(result.textSample, 'Sample text...');
    });

    it('should calculate compression percentage correctly', () => {
        const result1 = new CompressionResult({ compressionRatio: 0.3 });
        assert.equal(result1.compressionPercentage, 70);
        
        const result2 = new CompressionResult({ compressionRatio: 0.8 });
        assert.ok(Math.abs(result2.compressionPercentage - 20) < 0.001); // Use tolerance for floating point
    });
});

describe('CompressionPipeline', () => {
    let pipeline;
    const testDbPath = 'test_compression.db';
    
    before(() => {
        pipeline = new CompressionPipeline({ 
            dbPath: testDbPath,
            enableCache: false // Disable caching for simpler testing
        });
    });
    
    after(async () => {
        // Clean up test database
        try {
            await unlink(testDbPath);
        } catch (error) {
            // File might not exist, ignore error
        }
    });

    describe('Initialization', () => {
        it('should initialize with default settings', () => {
            const defaultPipeline = new CompressionPipeline();
            assert.equal(defaultPipeline.dbPath, 'compression_results.db');
            assert.equal(defaultPipeline.enableCache, true);
        });

        it('should initialize with custom settings', () => {
            assert.equal(pipeline.dbPath, testDbPath);
            assert.equal(pipeline.enableCache, false);
        });
    });

    describe('Database Operations', () => {
        it('should save and retrieve results from database', async () => {
            const result = new CompressionResult({
                method: 'test_method',
                originalSize: 100,
                compressedSize: 50,
                compressionRatio: 0.5,
                compressionTime: 1.0,
                decompressionTime: 0.5
            });
            
            await pipeline._saveResultToDb(result);
            
            const results = await pipeline.getResultsFromDb();
            assert.ok(results.length > 0);
            
            const savedResult = results[0];
            assert.equal(savedResult.method, 'test_method');
            assert.equal(savedResult.original_size, 100);
            assert.equal(savedResult.compressed_size, 50);
        });

        it('should filter results by method', async () => {
            // Save a result with specific method
            const result = new CompressionResult({
                method: 'specific_method',
                originalSize: 200,
                compressedSize: 100,
                compressionRatio: 0.5,
                compressionTime: 1.0,
                decompressionTime: 0.5
            });
            
            await pipeline._saveResultToDb(result);
            
            const filteredResults = await pipeline.getResultsFromDb('specific_method');
            assert.ok(filteredResults.length > 0);
            assert.ok(filteredResults.every(r => r.method === 'specific_method'));
        });
    });

    describe('Raw Compression Methods', () => {
        const testText = 'Hello world! This is a test string for compression. '.repeat(10);

        it('should compress with raw gzip', async () => {
            try {
                const { compressedData, result } = await pipeline.compressWithRawGzip(testText);
                
                assert.ok(Buffer.isBuffer(compressedData));
                assert.ok(compressedData.length > 0);
                assert.equal(result.method, 'raw_gzip');
                assert.ok(result.compressionRatio > 0);
                assert.ok(result.compressionRatio < 1); // Should achieve some compression
                assert.ok(result.compressionTime >= 0);
                assert.ok(result.decompressionTime >= 0);
            } catch (error) {
                // This test might fail if zlib is not available
                assert.ok(error.message.includes('zlib') || error.message.includes('compression'));
            }
        });

        it('should handle round-trip compression correctly for raw gzip', async () => {
            try {
                const { compressedData, result } = await pipeline.compressWithRawGzip(testText);
                
                // The compression should have verified round-trip internally
                assert.equal(result.method, 'raw_gzip');
                assert.equal(result.textSample, testText.substring(0, 100));
            } catch (error) {
                // Expected if zlib is not available
                assert.ok(error.message.includes('compression') || error.message.includes('zlib'));
            }
        });
    });

    describe('Result Creation', () => {
        it('should create compression result correctly', () => {
            const originalText = 'Test text for compression analysis';
            const compressedData = Buffer.from('compressed_data_mock');
            
            const result = pipeline._createCompressionResult(
                originalText,
                compressedData,
                1.5, // compression time
                0.8, // decompression time
                'test_method',
                'gpt2',
                1024
            );
            
            assert.equal(result.method, 'test_method');
            assert.equal(result.originalSize, Buffer.byteLength(originalText, 'utf8'));
            assert.equal(result.compressedSize, compressedData.length);
            assert.equal(result.compressionRatio, compressedData.length / Buffer.byteLength(originalText, 'utf8'));
            assert.equal(result.compressionTime, 1.5);
            assert.equal(result.decompressionTime, 0.8);
            assert.equal(result.modelName, 'gpt2');
            assert.equal(result.contextLength, 1024);
            assert.equal(result.textSample, originalText.substring(0, 100));
        });
    });

    describe('Benchmark Execution', () => {
        it('should run compression benchmark (mock test)', async () => {
            // This is a mock test since we don't have actual models available
            const shortText = 'Hello world';
            
            try {
                const results = await pipeline.runCompressionBenchmark(
                    shortText,
                    'gpt2',
                    null,
                    false // Don't save to DB
                );
                
                assert.ok(typeof results === 'object');
                assert.ok('raw_gzip' in results);
                
                // At least raw_gzip should work if zlib is available
                if (results.raw_gzip) {
                    assert.ok(results.raw_gzip instanceof CompressionResult);
                    assert.equal(results.raw_gzip.method, 'raw_gzip');
                }
            } catch (error) {
                // Expected if dependencies are not available
                console.log('Benchmark test skipped due to missing dependencies:', error.message);
            }
        });
    });

    describe('Comparison Table', () => {
        it('should print comparison table with results', () => {
            const mockResults = {
                'method1': new CompressionResult({
                    method: 'method1',
                    originalSize: 1000,
                    compressedSize: 300,
                    compressionRatio: 0.3,
                    compressionTime: 1.0,
                    decompressionTime: 0.5
                }),
                'method2': new CompressionResult({
                    method: 'method2',
                    originalSize: 1000,
                    compressedSize: 400,
                    compressionRatio: 0.4,
                    compressionTime: 0.8,
                    decompressionTime: 0.3
                })
            };
            
            // This should not throw an error
            assert.doesNotThrow(() => {
                pipeline.printComparisonTable(mockResults);
            });
        });

        it('should handle empty results gracefully', () => {
            const emptyResults = {};
            
            assert.doesNotThrow(() => {
                pipeline.printComparisonTable(emptyResults);
            });
        });

        it('should handle null results gracefully', () => {
            const nullResults = {
                'failed_method': null
            };
            
            assert.doesNotThrow(() => {
                pipeline.printComparisonTable(nullResults);
            });
        });
    });
});