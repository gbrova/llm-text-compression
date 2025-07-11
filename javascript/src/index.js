import { CompressionPipeline, CompressionResult } from './compression-pipeline.js';
import { LLMRanker, rankTokens } from './llm-ranker.js';
import { HuffmanRankCompressor } from './huffman-compression.js';
import { RankerCache } from './ranker-cache.js';

/**
 * Main entry point for LLM text compression library
 */

// Export main classes and functions
export {
    CompressionPipeline,
    CompressionResult,
    LLMRanker,
    HuffmanRankCompressor,
    RankerCache,
    rankTokens
};

/**
 * Quick demonstration of the compression pipeline
 */
async function demonstrate() {
    console.log('🚀 LLM Text Compression - JavaScript Port');
    console.log('==========================================\n');
    
    // Create pipeline instance
    const pipeline = new CompressionPipeline({
        enableCache: true
    });
    
    // Sample texts of different lengths
    const sampleTexts = [
        'Hello, world!',
        'The quick brown fox jumps over the lazy dog. ' +
        'This sentence contains every letter of the alphabet at least once.',
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' +
        'Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ' +
        'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. ' +
        'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. ' +
        'Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'
    ];
    
    for (let i = 0; i < sampleTexts.length; i++) {
        const text = sampleTexts[i];
        console.log(`\n📝 Sample ${i + 1} (${text.length} characters):`);
        const displayText = text.length > 100 ? text.substring(0, 100) + "..." : text; console.log(`"${displayText}"`);
        
        try {
            // Test raw compression (most likely to work)
            console.log('\n⚡ Testing raw gzip compression...');
            const { compressedData, result } = await pipeline.compressWithRawGzip(text);
            
            console.log(`Original size: ${result.originalSize} bytes`);
            console.log(`Compressed size: ${result.compressedSize} bytes`);
            console.log(`Compression ratio: ${result.compressionRatio.toFixed(3)}`);
            console.log(`Space reduction: ${result.compressionPercentage.toFixed(1)}%`);
            console.log(`Compression time: ${result.compressionTime.toFixed(3)}s`);
            console.log(`Decompression time: ${result.decompressionTime.toFixed(3)}s`);
            
            // If we have a longer text, try to run a fuller benchmark
            if (text.length > 200) {
                console.log('\n🏃 Running full compression benchmark...');
                const results = await pipeline.runCompressionBenchmark(text, 'gpt2', null, false);
                pipeline.printComparisonTable(results);
            }
            
        } catch (error) {
            console.log(`❌ Error processing sample ${i + 1}: ${error.message}`);
        }
    }
    
    console.log('\n✅ Demonstration complete!');
    console.log('\nTo run more comprehensive benchmarks, use: npm run benchmark');
    console.log('To run tests, use: npm test');
}

/**
 * Example usage of individual components
 */
async function demonstrateComponents() {
    console.log('\n🔧 Component Demonstrations');
    console.log('============================\n');
    
    // Demonstrate LLM Ranker
    console.log('1. LLM Ranker:');
    try {
        const ranker = new LLMRanker();
        console.log(`   Model: ${ranker.modelName}`);
        console.log(`   Max context: ${ranker.getContextLength()}`);
        console.log(`   Batch size: ${ranker.batchSize}`);
        console.log(`   Cache signature: ${JSON.stringify(ranker.getCacheSignature())}`);
        
        // These would work if transformers.js is properly set up
        // const ranks = await ranker.getTokenRanks('Hello world');
        // console.log(`   Sample ranks: ${ranks.slice(0, 5)}`);
    } catch (error) {
        console.log(`   ⚠️  LLM Ranker demo skipped: ${error.message}`);
    }
    
    // Demonstrate Huffman Compressor
    console.log('\n2. Huffman Compressor:');
    try {
        const compressor = new HuffmanRankCompressor();
        const sampleRanks = [1, 2, 1, 3, 2, 1, 4, 1, 2, 3, 1, 5];
        
        console.log(`   Sample ranks: ${sampleRanks}`);
        console.log(`   Rank frequencies:`, compressor._countRankFrequencies(sampleRanks));
        
        // Test Zipf fitting
        const rankCounts = compressor._countRankFrequencies(sampleRanks);
        const { sParam, maxRank } = compressor._fitZipfDistribution(rankCounts);
        console.log(`   Fitted Zipf parameter: ${sParam.toFixed(3)}`);
        console.log(`   Max rank: ${maxRank}`);
        
        const zipfFreqs = compressor._generateZipfFrequencies(sParam, maxRank, sampleRanks.length);
        console.log(`   Generated Zipf frequencies:`, zipfFreqs);
        
    } catch (error) {
        console.log(`   ⚠️  Huffman Compressor demo limited: ${error.message}`);
    }
    
    // Demonstrate Ranker Cache
    console.log('\n3. Ranker Cache:');
    try {
        const cache = new RankerCache('demo_cache.db');
        const stats = await cache.getCacheStats();
        console.log(`   Cache statistics:`, stats);
    } catch (error) {
        console.log(`   ⚠️  Ranker Cache demo skipped: ${error.message}`);
    }
}

// Run demonstration if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    try {
        await demonstrate();
        await demonstrateComponents();
    } catch (error) {
        console.error('Demonstration failed:', error);
        process.exit(1);
    }
}