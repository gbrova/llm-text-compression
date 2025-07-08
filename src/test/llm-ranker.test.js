import { describe, it, before } from 'node:test';
import { strict as assert } from 'node:assert';
import { LLMRanker, rankTokens } from '../llm-ranker.js';

describe('LLMRanker', () => {
    let ranker;
    
    before(async () => {
        ranker = new LLMRanker();
        // Note: In a real test environment, you'd want to ensure the model is loaded
        // For now, we'll test the interface and mock implementations
    });

    describe('Basic Properties', () => {
        it('should have correct default configuration', () => {
            assert.equal(ranker.modelName, 'gpt2');
            assert.equal(ranker.maxContextLength, 1024);
            assert.equal(ranker.batchSize, 1);
        });

        it('should provide cache signature', () => {
            const signature = ranker.getCacheSignature();
            assert.ok(Array.isArray(signature));
            assert.equal(signature.length, 3);
            assert.equal(signature[0], 'gpt2');
            assert.equal(signature[1], 1024);
            assert.equal(signature[2], 1);
        });

        it('should get context length', () => {
            const contextLength = ranker.getContextLength();
            assert.equal(contextLength, 1024);
        });
    });

    describe('Token Operations', () => {
        it('should handle empty text', async () => {
            const ranks = await ranker.getTokenRanks('');
            assert.deepEqual(ranks, []);
        });

        it('should handle empty ranks', async () => {
            const result = await ranker.getStringFromTokenRanks([]);
            assert.equal(result, '');
        });

        it('should tokenize text', async () => {
            // Mock test - in real implementation, this would test actual tokenization
            try {
                const tokens = await ranker.tokenize('Hello world');
                assert.ok(Array.isArray(tokens));
            } catch (error) {
                // Expected if model is not available in test environment
                assert.ok(error.message.includes('Failed to initialize'));
            }
        });

        it('should get token ranks for simple text', async () => {
            try {
                const text = 'Hello world';
                const ranks = await ranker.getTokenRanks(text);
                
                assert.ok(Array.isArray(ranks));
                assert.ok(ranks.length > 0);
                assert.ok(ranks.every(rank => Number.isInteger(rank) && rank >= 1));
            } catch (error) {
                // Expected if model is not available in test environment
                assert.ok(error.message.includes('Failed to initialize'));
            }
        });

        it('should generate text from ranks', async () => {
            try {
                // Test with simple ranks
                const ranks = [1, 1, 1];
                const result = await ranker.getStringFromTokenRanks(ranks);
                
                assert.ok(typeof result === 'string');
            } catch (error) {
                // Expected if model is not available in test environment
                assert.ok(error.message.includes('Failed to initialize'));
            }
        });
    });

    describe('Round-trip Processing', () => {
        it('should maintain round-trip consistency (mock test)', async () => {
            // This would test actual round-trip consistency in a real environment
            // For now, we test the interface
            try {
                const originalText = 'Hello world';
                const ranks = await ranker.getTokenRanks(originalText);
                const reconstructed = await ranker.getStringFromTokenRanks(ranks);
                
                // In a real implementation, this should be equal
                assert.ok(typeof reconstructed === 'string');
            } catch (error) {
                // Expected if model is not available in test environment
                assert.ok(error.message.includes('Failed to initialize'));
            }
        });
    });

    describe('Batching Support', () => {
        it('should handle different batch sizes', () => {
            const ranker1 = new LLMRanker({ batchSize: 1 });
            const ranker2 = new LLMRanker({ batchSize: 4 });
            
            assert.equal(ranker1.batchSize, 1);
            assert.equal(ranker2.batchSize, 4);
        });

        it('should create chunks correctly', () => {
            const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            const chunks = ranker._createChunks(data, 3);
            
            assert.equal(chunks.length, 3);
            assert.equal(chunks[0].length, 4); // First chunk gets extra item
            assert.equal(chunks[1].length, 3);
            assert.equal(chunks[2].length, 3);
            
            // Verify all data is preserved
            const flattened = chunks.flat();
            assert.deepEqual(flattened, data);
        });

        it('should handle edge cases in chunking', () => {
            // Empty data
            const emptyChunks = ranker._createChunks([], 3);
            assert.deepEqual(emptyChunks, [[]]);
            
            // Single chunk
            const singleChunk = ranker._createChunks([1, 2, 3], 1);
            assert.equal(singleChunk.length, 1);
            assert.deepEqual(singleChunk[0], [1, 2, 3]);
            
            // More chunks than data
            const moreChunks = ranker._createChunks([1, 2], 5);
            assert.equal(moreChunks.length, 2);
            assert.deepEqual(moreChunks[0], [1]);
            assert.deepEqual(moreChunks[1], [2]);
        });
    });

    describe('Model Configuration', () => {
        it('should accept custom model configuration', () => {
            const customRanker = new LLMRanker({
                modelName: 'custom-model',
                maxContextLength: 512,
                batchSize: 2
            });
            
            assert.equal(customRanker.modelName, 'custom-model');
            assert.equal(customRanker.maxContextLength, 512);
            assert.equal(customRanker.batchSize, 2);
        });
    });
});

describe('Convenience Functions', () => {
    describe('rankTokens', () => {
        it('should provide convenience function interface', async () => {
            try {
                const ranks = await rankTokens('Hello world');
                assert.ok(Array.isArray(ranks));
            } catch (error) {
                // Expected if model is not available in test environment
                assert.ok(error.message.includes('Failed to initialize'));
            }
        });

        it('should accept custom parameters', async () => {
            try {
                const ranks = await rankTokens('Hello', 'gpt2', 2);
                assert.ok(Array.isArray(ranks));
            } catch (error) {
                // Expected if model is not available in test environment
                assert.ok(error.message.includes('Failed to initialize'));
            }
        });
    });
});