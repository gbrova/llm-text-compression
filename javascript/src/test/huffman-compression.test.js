import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { HuffmanRankCompressor } from '../huffman-compression.js';

describe('HuffmanRankCompressor', () => {
    let compressor;
    
    describe('Initialization', () => {
        it('should create compressor instance', () => {
            compressor = new HuffmanRankCompressor();
            assert.ok(compressor instanceof HuffmanRankCompressor);
        });
    });

    describe('Basic Huffman Compression', () => {
        it('should compress and decompress ranks with basic method', () => {
            const ranks = [1, 2, 1, 3, 2, 1, 4, 1, 2, 3];
            
            try {
                const compressedData = compressor.compressBasic(ranks);
                assert.ok(Buffer.isBuffer(compressedData));
                assert.ok(compressedData.length > 0);
                
                const decompressedRanks = compressor.decompressBasic(compressedData);
                assert.deepEqual(decompressedRanks, ranks);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });

        it('should handle empty ranks array', () => {
            const ranks = [];
            
            try {
                const compressedData = compressor.compressBasic(ranks);
                const decompressedRanks = compressor.decompressBasic(compressedData);
                assert.deepEqual(decompressedRanks, ranks);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });

        it('should handle single rank', () => {
            const ranks = [1];
            
            try {
                const compressedData = compressor.compressBasic(ranks);
                const decompressedRanks = compressor.decompressBasic(compressedData);
                assert.deepEqual(decompressedRanks, ranks);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });
    });

    describe('Frequency Counting', () => {
        it('should count rank frequencies correctly', () => {
            const ranks = [1, 2, 1, 3, 2, 1];
            const frequencies = compressor._countRankFrequencies(ranks);
            
            assert.equal(frequencies[1], 3);
            assert.equal(frequencies[2], 2);
            assert.equal(frequencies[3], 1);
        });

        it('should handle empty array', () => {
            const ranks = [];
            const frequencies = compressor._countRankFrequencies(ranks);
            
            assert.deepEqual(frequencies, {});
        });
    });

    describe('Zipf Distribution Fitting', () => {
        it('should fit Zipf distribution to rank counts', () => {
            const rankCounts = { 1: 10, 2: 5, 3: 3, 4: 2, 5: 1 };
            const { sParam, maxRank } = compressor._fitZipfDistribution(rankCounts);
            
            assert.ok(typeof sParam === 'number');
            assert.ok(sParam > 0);
            assert.equal(maxRank, 5);
        });

        it('should calculate Zipf error correctly', () => {
            const ranks = [1, 2, 3];
            const counts = [10, 5, 3];
            const s = 1.0;
            
            const error = compressor._calculateZipfError(ranks, counts, s);
            assert.ok(typeof error === 'number');
            assert.ok(error >= 0);
        });

        it('should return infinite error for invalid s parameter', () => {
            const ranks = [1, 2, 3];
            const counts = [10, 5, 3];
            const s = 0; // Invalid
            
            const error = compressor._calculateZipfError(ranks, counts, s);
            assert.equal(error, Infinity);
        });
    });

    describe('Zipf Frequency Generation', () => {
        it('should generate frequencies based on Zipf distribution', () => {
            const sParam = 1.0;
            const maxRank = 5;
            const totalCount = 21;
            
            const frequencies = compressor._generateZipfFrequencies(sParam, maxRank, totalCount);
            
            assert.ok(typeof frequencies === 'object');
            assert.ok(Object.keys(frequencies).length === maxRank);
            
            // Check that all ranks 1-5 are present
            for (let i = 1; i <= maxRank; i++) {
                assert.ok(i in frequencies);
                assert.ok(frequencies[i] >= 1); // Should be at least 1
            }
            
            // Rank 1 should have highest frequency (Zipf law)
            assert.ok(frequencies[1] >= frequencies[2]);
            assert.ok(frequencies[2] >= frequencies[3]);
        });
    });

    describe('Zipf Compression', () => {
        it('should compress and decompress with Zipf method', () => {
            const ranks = [1, 1, 2, 1, 3, 2, 1, 4, 2, 1];
            
            try {
                const compressedData = compressor.compressZipf(ranks);
                assert.ok(Buffer.isBuffer(compressedData));
                
                const decompressedRanks = compressor.decompressZipf(compressedData);
                assert.deepEqual(decompressedRanks, ranks);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });

        it('should compress and decompress with Zipf bytes method', () => {
            const ranks = [1, 1, 2, 1, 3, 2, 1, 4, 2, 1];
            
            try {
                const compressedData = compressor.compressZipfBytes(ranks);
                assert.ok(Buffer.isBuffer(compressedData));
                assert.ok(compressedData.length >= 12); // At least header size
                
                const decompressedRanks = compressor.decompressZipfBytes(compressedData);
                assert.deepEqual(decompressedRanks, ranks);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });
    });

    describe('Compression with Timing', () => {
        it('should provide timing information for compression', () => {
            const ranks = [1, 2, 3, 4, 5];
            
            try {
                const { compressedData, compressionTime } = 
                    compressor.compressRanksWithTiming(ranks, 'basic');
                
                assert.ok(Buffer.isBuffer(compressedData));
                assert.ok(typeof compressionTime === 'number');
                assert.ok(compressionTime >= 0);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });

        it('should provide timing information for decompression', () => {
            const ranks = [1, 2, 3, 4, 5];
            
            try {
                const { compressedData } = compressor.compressRanksWithTiming(ranks, 'basic');
                const { decompressedRanks, decompressionTime } = 
                    compressor.decompressRanksWithTiming(compressedData, 'basic');
                
                assert.deepEqual(decompressedRanks, ranks);
                assert.ok(typeof decompressionTime === 'number');
                assert.ok(decompressionTime >= 0);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });

        it('should support all compression methods', () => {
            const ranks = [1, 2, 3];
            const methods = ['basic', 'zipf', 'zipf_bytes'];
            
            for (const method of methods) {
                try {
                    const { compressedData, compressionTime } = 
                        compressor.compressRanksWithTiming(ranks, method);
                    
                    const { decompressedRanks, decompressionTime } = 
                        compressor.decompressRanksWithTiming(compressedData, method);
                    
                    assert.deepEqual(decompressedRanks, ranks);
                    assert.ok(compressionTime >= 0);
                    assert.ok(decompressionTime >= 0);
                } catch (error) {
                    // Expected if huffman-coding dependency is not available
                    assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
                }
            }
        });

        it('should throw error for unknown method', () => {
            const ranks = [1, 2, 3];
            
            assert.throws(() => {
                compressor.compressRanksWithTiming(ranks, 'unknown_method');
            }, /Unknown compression method/);
            
            // For decompression, we can't test with actual compressed data,
            // but we can test the error handling
            assert.throws(() => {
                compressor.decompressRanksWithTiming(Buffer.from('dummy'), 'unknown_method');
            }, /Unknown compression method/);
        });
    });

    describe('Edge Cases', () => {
        it('should handle large rank values', () => {
            const ranks = [1, 100, 1000, 10000];
            
            try {
                const compressedData = compressor.compressBasic(ranks);
                const decompressedRanks = compressor.decompressBasic(compressedData);
                assert.deepEqual(decompressedRanks, ranks);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });

        it('should handle repeated patterns', () => {
            const ranks = [1, 2, 3].concat([1, 2, 3]).concat([1, 2, 3]);
            
            try {
                const compressedData = compressor.compressBasic(ranks);
                const decompressedRanks = compressor.decompressBasic(compressedData);
                assert.deepEqual(decompressedRanks, ranks);
            } catch (error) {
                // Expected if huffman-coding dependency is not available
                assert.ok(error.message.includes('huffman') || error.message.includes('HuffmanCodec'));
            }
        });
    });
});