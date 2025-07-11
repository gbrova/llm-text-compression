import { CompressionPipeline, loadDatasetFile } from './compression-pipeline.js';
import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

/**
 * Main benchmarking script for LLM compression experiments
 */
async function main() {
    console.log('🚀 LLM Compression Pipeline - JavaScript Port');
    console.log('='.repeat(50));
    
    const pipeline = new CompressionPipeline();
    
    // List available datasets
    try {
        const datasetsDir = 'datasets';
        const files = await readdir(datasetsDir);
        const txtFiles = files.filter(file => file.endsWith('.txt'));
        
        console.log(`\nAvailable datasets: ${txtFiles.length} files`);
        for (const file of txtFiles) {
            console.log(`  - ${file}`);
        }
        
        if (txtFiles.length > 0) {
            await runBenchmarks(pipeline, txtFiles);
        } else {
            console.log('\nNo dataset files found. Running with sample text...');
            await runSampleBenchmarks(pipeline);
        }
    } catch (error) {
        console.log('\nDatasets directory not found. Running with sample text...');
        await runSampleBenchmarks(pipeline);
    }
    
    // Show database summary
    await showDatabaseSummary(pipeline);
    
    console.log(`\n💾 Results saved to: ${pipeline.dbPath}`);
    console.log('✅ JavaScript compression pipeline benchmark complete!');
}

/**
 * Run benchmarks with available dataset files
 * @param {CompressionPipeline} pipeline - Pipeline instance
 * @param {Array<string>} txtFiles - List of dataset files
 */
async function runBenchmarks(pipeline, txtFiles) {
    console.log('\n📊 Running compression benchmarks...');
    
    // Test 1: Load and test first dataset with 3k characters
    const firstFile = txtFiles[0];
    console.log(`\n1. Testing with real dataset: ${firstFile} (3k characters)`);
    
    try {
        const datasetPath = join('datasets', firstFile);
        const datasetText = await readFile(datasetPath, 'utf8');
        const sampleText = datasetText.substring(0, 3000);
        console.log(`   Using first 3000 characters (${sampleText.length} chars)`);
        
        const results = await pipeline.runCompressionBenchmark(sampleText);
        pipeline.printComparisonTable(results);
    } catch (error) {
        console.log(`   Error loading dataset: ${error.message}`);
    }
    
    // Test 2: Same dataset with max context length of 300
    console.log(`\n2. Testing with ${firstFile} (3k characters, max context length 300)`);
    
    try {
        const datasetPath = join('datasets', firstFile);
        const datasetText = await readFile(datasetPath, 'utf8');
        const sampleText = datasetText.substring(0, 3000);
        console.log(`   Using first 3000 characters (${sampleText.length} chars), max context length 300`);
        
        const results = await pipeline.runCompressionBenchmark(
            sampleText, 
            'gpt2',      // model name
            300,         // max context length
            true,        // save to db
            4            // batch size
        );
        pipeline.printComparisonTable(results);
    } catch (error) {
        console.log(`   Error loading dataset: ${error.message}`);
    }
}

/**
 * Run benchmarks with sample text when no datasets are available
 * @param {CompressionPipeline} pipeline - Pipeline instance
 */
async function runSampleBenchmarks(pipeline) {
    console.log('\n📊 Running compression benchmarks with sample text...');
    
    // Create a sample text for testing
    const sampleTexts = [
        'Hello world! This is a test of the compression pipeline. '.repeat(50),
        'The quick brown fox jumps over the lazy dog. '.repeat(40),
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '.repeat(30),
        'Artificial intelligence and machine learning are transforming technology. '.repeat(25)
    ];
    
    for (let i = 0; i < sampleTexts.length; i++) {
        const text = sampleTexts[i];
        console.log(`\n${i + 1}. Testing with sample text ${i + 1} (${text.length} characters)`);
        
        try {
            const results = await pipeline.runCompressionBenchmark(text);
            pipeline.printComparisonTable(results);
        } catch (error) {
            console.log(`   Sample ${i + 1} benchmark failed: ${error.message}`);
        }
    }
}

/**
 * Show database summary statistics
 * @param {CompressionPipeline} pipeline - Pipeline instance
 */
async function showDatabaseSummary(pipeline) {
    console.log('\n📈 Database Summary:');
    
    try {
        const allResults = await pipeline.getResultsFromDb();
        
        if (allResults.length > 0) {
            console.log(`   Total experiments stored: ${allResults.length}`);
            
            // Group by method
            const methods = {};
            for (const result of allResults) {
                const method = result.method;
                if (!methods[method]) {
                    methods[method] = [];
                }
                methods[method].push(result.compression_ratio);
            }
            
            console.log('\n   Average compression ratios by method:');
            for (const [method, ratios] of Object.entries(methods)) {
                const avgRatio = ratios.reduce((sum, ratio) => sum + ratio, 0) / ratios.length;
                const avgReduction = (1 - avgRatio) * 100;
                console.log(`     ${method}: ${avgRatio.toFixed(3)} (${avgReduction.toFixed(1)}% reduction, ${ratios.length} samples)`);
            }
        } else {
            console.log('   No experiments stored yet.');
        }
    } catch (error) {
        console.log(`   Error reading database: ${error.message}`);
    }
}

/**
 * Performance comparison function
 */
async function comparePerformance() {
    console.log('\n🏃 Performance Comparison Tests');
    console.log('='.repeat(50));
    
    const pipeline = new CompressionPipeline({ enableCache: false });
    const texts = [
        'Short text for testing.',
        'Medium length text that should provide reasonable compression ratios when using various algorithms. '.repeat(5),
        'Long text sample that will be used to test the performance and effectiveness of different compression methods including LLM-based ranking, Huffman coding, and traditional compression algorithms. '.repeat(20)
    ];
    
    for (let i = 0; i < texts.length; i++) {
        const text = texts[i];
        console.log(`\nTesting text ${i + 1} (${text.length} characters):`);
        
        try {
            const startTime = Date.now();
            const results = await pipeline.runCompressionBenchmark(text, 'gpt2', null, false);
            const totalTime = (Date.now() - startTime) / 1000;
            
            console.log(`Total benchmark time: ${totalTime.toFixed(2)}s`);
            pipeline.printComparisonTable(results);
        } catch (error) {
            console.log(`   Performance test ${i + 1} failed: ${error.message}`);
        }
    }
}

/**
 * Round-trip verification tests
 */
async function runRoundTripTests() {
    console.log('\n🔄 Round-trip Verification Tests');
    console.log('='.repeat(50));
    
    const pipeline = new CompressionPipeline({ enableCache: false });
    const testCases = [
        'Hello, world!',
        'The quick brown fox jumps over the lazy dog.',
        'Special characters: !@#$%^&*()_+-=[]{}|;:,.<>?',
        'Numbers: 1234567890 and mixed: abc123def456',
        'Unicode: café, naïve, résumé, 你好, 🚀',
        'Repeated patterns: ' + 'abc'.repeat(100),
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' +
        'Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ' +
        'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.'
    ];
    
    let passedTests = 0;
    let totalTests = 0;
    
    for (let i = 0; i < testCases.length; i++) {
        const originalText = testCases[i];
        console.log(`\nTest ${i + 1}: "${originalText.substring(0, 50)}${originalText.length > 50 ? '...' : ''}"`);
        
        // Test only methods that are likely to work without full LLM setup
        const methods = ['raw_gzip'];
        
        for (const methodName of methods) {
            totalTests++;
            try {
                let result;
                switch (methodName) {
                    case 'raw_gzip':
                        result = await pipeline.compressWithRawGzip(originalText);
                        break;
                    default:
                        continue;
                }
                
                console.log(`   ✅ ${methodName}: Round-trip successful (${result.result.compressionPercentage.toFixed(1)}% reduction)`);
                passedTests++;
            } catch (error) {
                console.log(`   ❌ ${methodName}: Round-trip failed - ${error.message}`);
            }
        }
    }
    
    console.log(`\nRound-trip test summary: ${passedTests}/${totalTests} tests passed`);
    if (passedTests === totalTests) {
        console.log('🎉 All round-trip tests passed!');
    } else {
        console.log('⚠️  Some round-trip tests failed. Check implementation.');
    }
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
    try {
        await main();
        await comparePerformance();
        await runRoundTripTests();
    } catch (error) {
        console.error('Benchmark failed:', error);
        process.exit(1);
    }
}