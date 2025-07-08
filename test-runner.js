#!/usr/bin/env node

/**
 * Simple test runner to verify core functionality without external dependencies
 */

import { CompressionResult } from './src/compression-pipeline.js';
import { LLMRanker } from './src/llm-ranker.js';
import { HuffmanRankCompressor } from './src/huffman-compression.js';

console.log('🧪 Running JavaScript Port Tests');
console.log('================================\n');

let passed = 0;
let failed = 0;

function test(description, testFn) {
    try {
        testFn();
        console.log(`✅ ${description}`);
        passed++;
    } catch (error) {
        console.log(`❌ ${description}: ${error.message}`);
        failed++;
    }
}

function asyncTest(description, testFn) {
    return testFn().then(() => {
        console.log(`✅ ${description}`);
        passed++;
    }).catch(error => {
        console.log(`❌ ${description}: ${error.message}`);
        failed++;
    });
}

// Test CompressionResult
console.log('📊 Testing CompressionResult...');
test('should create compression result', () => {
    const result = new CompressionResult({
        method: 'test',
        originalSize: 1000,
        compressedSize: 500,
        compressionRatio: 0.5
    });
    
    if (result.method !== 'test') throw new Error('Method not set correctly');
    if (result.originalSize !== 1000) throw new Error('Original size not set correctly');
    if (Math.abs(result.compressionPercentage - 50) > 0.001) throw new Error('Compression percentage calculation failed');
});

// Test LLMRanker
console.log('\n🤖 Testing LLMRanker...');
test('should create ranker with default config', () => {
    const ranker = new LLMRanker();
    if (ranker.modelName !== 'gpt2') throw new Error('Default model name incorrect');
    if (ranker.maxContextLength !== 1024) throw new Error('Default context length incorrect');
    if (ranker.batchSize !== 1) throw new Error('Default batch size incorrect');
});

test('should create ranker with custom config', () => {
    const ranker = new LLMRanker({
        modelName: 'custom-model',
        maxContextLength: 512,
        batchSize: 4
    });
    if (ranker.modelName !== 'custom-model') throw new Error('Custom model name incorrect');
    if (ranker.maxContextLength !== 512) throw new Error('Custom context length incorrect');
    if (ranker.batchSize !== 4) throw new Error('Custom batch size incorrect');
});

test('should provide cache signature', () => {
    const ranker = new LLMRanker();
    const signature = ranker.getCacheSignature();
    if (!Array.isArray(signature)) throw new Error('Cache signature should be array');
    if (signature.length !== 3) throw new Error('Cache signature should have 3 elements');
});

test('should create chunks correctly', () => {
    const ranker = new LLMRanker();
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const chunks = ranker._createChunks(data, 3);
    
    if (chunks.length !== 3) throw new Error('Should create 3 chunks');
    if (chunks[0].length !== 4) throw new Error('First chunk should have 4 items');
    if (chunks[1].length !== 3) throw new Error('Second chunk should have 3 items');
    if (chunks[2].length !== 3) throw new Error('Third chunk should have 3 items');
    
    // Verify all data is preserved
    const flattened = chunks.flat();
    if (JSON.stringify(flattened) !== JSON.stringify(data)) {
        throw new Error('Data not preserved in chunking');
    }
});

// Test HuffmanRankCompressor
console.log('\n🗜️  Testing HuffmanRankCompressor...');
test('should create compressor', () => {
    const compressor = new HuffmanRankCompressor();
    if (!(compressor instanceof HuffmanRankCompressor)) {
        throw new Error('Should create compressor instance');
    }
});

test('should count rank frequencies', () => {
    const compressor = new HuffmanRankCompressor();
    const ranks = [1, 2, 1, 3, 2, 1];
    const frequencies = compressor._countRankFrequencies(ranks);
    
    if (frequencies[1] !== 3) throw new Error('Frequency count for rank 1 incorrect');
    if (frequencies[2] !== 2) throw new Error('Frequency count for rank 2 incorrect');
    if (frequencies[3] !== 1) throw new Error('Frequency count for rank 3 incorrect');
});

test('should fit Zipf distribution', () => {
    const compressor = new HuffmanRankCompressor();
    const rankCounts = { 1: 10, 2: 5, 3: 3, 4: 2, 5: 1 };
    const { sParam, maxRank } = compressor._fitZipfDistribution(rankCounts);
    
    if (typeof sParam !== 'number' || sParam <= 0) {
        throw new Error('Zipf parameter should be positive number');
    }
    if (maxRank !== 5) throw new Error('Max rank should be 5');
});

test('should generate Zipf frequencies', () => {
    const compressor = new HuffmanRankCompressor();
    const frequencies = compressor._generateZipfFrequencies(1.0, 5, 21);
    
    if (Object.keys(frequencies).length !== 5) {
        throw new Error('Should generate frequencies for all ranks');
    }
    if (frequencies[1] < frequencies[2]) {
        throw new Error('Rank 1 should have highest frequency (Zipf law)');
    }
});

// Test async functionality with mock implementations
console.log('\n⚡ Testing async functionality...');

await asyncTest('should handle empty token ranks', async () => {
    const ranker = new LLMRanker();
    const ranks = await ranker.getTokenRanks('');
    if (!Array.isArray(ranks)) throw new Error('Should return array');
    if (ranks.length !== 0) throw new Error('Empty text should return empty ranks');
});

await asyncTest('should handle empty string generation', async () => {
    const ranker = new LLMRanker();
    const result = await ranker.getStringFromTokenRanks([]);
    if (typeof result !== 'string') throw new Error('Should return string');
    if (result !== '') throw new Error('Empty ranks should return empty string');
});

await asyncTest('should tokenize text (mock)', async () => {
    const ranker = new LLMRanker();
    const tokens = await ranker.tokenize('Hello world');
    if (!Array.isArray(tokens)) throw new Error('Should return token array');
});

await asyncTest('should decode tokens (mock)', async () => {
    const ranker = new LLMRanker();
    const text = await ranker.decode([1, 2, 3]);
    if (typeof text !== 'string') throw new Error('Should return string');
});

// Summary
console.log('\n📋 Test Summary');
console.log('================');
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Total: ${passed + failed}`);

if (failed === 0) {
    console.log('\n🎉 All tests passed! The JavaScript port is working correctly.');
    console.log('Note: These tests use mock implementations for external dependencies.');
    console.log('Install the full dependencies to test with real LLM inference.');
} else {
    console.log(`\n⚠️  ${failed} test(s) failed. Check the implementation.`);
    process.exit(1);
}

console.log('\n📦 To install full dependencies:');
console.log('npm install @huggingface/transformers sqlite3 huffman-coding');

console.log('\n🚀 To run benchmarks:');
console.log('npm run benchmark');

console.log('\n🔬 To run full test suite:');
console.log('npm test');