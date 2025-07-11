#!/usr/bin/env node

/**
 * Test script for real round-trip functionality
 * This shows what round-trip testing should look like with a real LLM
 */

import { LLMRanker } from './src/llm-ranker.js';

console.log('🧪 Testing Real Round-trip Functionality');
console.log('==========================================\n');

const ranker = new LLMRanker({ modelName: 'Xenova/distilgpt2' });

async function testTokenization() {
    console.log('📝 Testing basic tokenization round-trip...');
    try {
        const text = 'Hello world';
        console.log(`Original text: "${text}"`);
        
        // Test tokenization
        const tokens = await ranker.tokenize(text);
        console.log('Tokens:', tokens);
        
        // Test detokenization  
        const decoded = await ranker.decode(tokens);
        console.log(`Decoded text: "${decoded}"`);
        
        const isRoundTripSuccessful = text === decoded;
        console.log(`✅ Tokenization round-trip successful: ${isRoundTripSuccessful}\n`);
        
        return isRoundTripSuccessful;
    } catch (error) {
        console.error('❌ Tokenization test failed:', error.message);
        return false;
    }
}

async function testRankingRoundTrip() {
    console.log('🎯 Testing token ranking round-trip...');
    try {
        const text = 'Hello';
        console.log(`Original text: "${text}"`);
        
        // Get token ranks
        const ranks = await ranker.getTokenRanks(text);
        console.log('Token ranks:', ranks);
        
        // Convert back to text
        const reconstructed = await ranker.getStringFromTokenRanks(ranks);
        console.log(`Reconstructed text: "${reconstructed}"`);
        
        const isRoundTripSuccessful = text === reconstructed;
        console.log(`✅ Ranking round-trip successful: ${isRoundTripSuccessful}\n`);
        
        if (!isRoundTripSuccessful) {
            console.log('📋 Analysis:');
            console.log(`  - Original length: ${text.length}`);
            console.log(`  - Reconstructed length: ${reconstructed.length}`);
            console.log(`  - Character-by-character comparison:`);
            for (let i = 0; i < Math.max(text.length, reconstructed.length); i++) {
                const orig = text[i] || '(end)';
                const recon = reconstructed[i] || '(end)';
                const match = orig === recon ? '✓' : '✗';
                console.log(`    [${i}] "${orig}" vs "${recon}" ${match}`);
            }
        }
        
        return isRoundTripSuccessful;
    } catch (error) {
        console.error('❌ Ranking round-trip test failed:', error.message);
        console.error('Full error:', error);
        return false;
    }
}

async function testLongerTextRoundTrip() {
    console.log('📝 Testing longer text round-trip...');
    try {
        const text = 'The quick brown fox jumps over the lazy dog';
        console.log(`Original text: "${text}"`);
        
        // Get token ranks
        const ranks = await ranker.getTokenRanks(text);
        console.log('Token ranks:', ranks);
        console.log(`Number of tokens: ${ranks.length}`);
        
        // Convert back to text
        const reconstructed = await ranker.getStringFromTokenRanks(ranks);
        console.log(`Reconstructed text: "${reconstructed}"`);
        
        const isRoundTripSuccessful = text === reconstructed;
        console.log(`✅ Longer text round-trip successful: ${isRoundTripSuccessful}\n`);
        
        if (!isRoundTripSuccessful) {
            console.log('📋 Analysis:');
            console.log(`  - Original length: ${text.length}`);
            console.log(`  - Reconstructed length: ${reconstructed.length}`);
            console.log(`  - Character-by-character comparison:`);
            for (let i = 0; i < Math.max(text.length, reconstructed.length); i++) {
                const orig = text[i] || '(end)';
                const recon = reconstructed[i] || '(end)';
                const match = orig === recon ? '✓' : '✗';
                console.log(`    [${i}] "${orig}" vs "${recon}" ${match}`);
                if (i > 20 && !match) {
                    console.log('    ... (truncated)');
                    break;
                }
            }
        }
        
        return isRoundTripSuccessful;
    } catch (error) {
        console.error('❌ Longer text round-trip test failed:', error.message);
        console.error('Full error:', error);
        return false;
    }
}

async function main() {
    console.log('🚀 Starting real LLM round-trip tests...\n');
    
    const tokenRoundTrip = await testTokenization();
    const rankRoundTrip = await testRankingRoundTrip();
    const longerTextRoundTrip = await testLongerTextRoundTrip();
    
    console.log('📊 Summary:');
    console.log(`  Tokenization round-trip: ${tokenRoundTrip ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`  Ranking round-trip: ${rankRoundTrip ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`  Longer text round-trip: ${longerTextRoundTrip ? '✅ PASS' : '❌ FAIL'}`);
    
    if (tokenRoundTrip && rankRoundTrip && longerTextRoundTrip) {
        console.log('\n🎉 All round-trip tests passed! The implementation preserves data correctly.');
    } else {
        console.log('\n⚠️  Some round-trip tests failed. This indicates the implementation may not be suitable for lossless compression.');
    }
    
    console.log('\n💡 Note: Round-trip correctness is essential for:');
    console.log('   • Lossless text compression');
    console.log('   • Accurate compression benchmarks');
    console.log('   • Reliable compression ratios');
}

main().catch(console.error);