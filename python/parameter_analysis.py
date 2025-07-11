"""
Parameter Analysis Script for LLM Compression

This script systematically varies each parameter to understand its impact on compression performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from compression_pipeline import CompressionPipeline, load_dataset_file
import time

def analyze_document_size():
    """Analyze how document size affects compression ratio and processing time."""
    print("📊 Analyzing Document Size Effects...")
    
    pipeline = CompressionPipeline(enable_cache=False)
    dataset_text = load_dataset_file("bbc-yachts.txt")
    
    # Test different document sizes (reduced for faster execution)
    sizes = [500, 1000, 2000, 3000]
    results = []
    
    for size in sizes:
        print(f"Testing size: {size} characters")
        sample_text = dataset_text[:size]
        
        # Test with our best method: Zipf-bytes Huffman
        try:
            start_time = time.time()
            compressed_data, result = pipeline.compress_with_llm_ranks_huffman_zipf_bytes(sample_text)
            processing_time = time.time() - start_time
            
            results.append({
                'size': size,
                'compression_ratio': result.compression_ratio,
                'compression_percentage': result.compression_percentage,
                'processing_time': processing_time,
                'compressed_size': len(compressed_data),
                'original_size': len(sample_text.encode('utf-8'))
            })
            
            print(f"  -> {result.compression_percentage:.1f}% reduction, {processing_time:.1f}s")
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    return results

def analyze_batch_size():
    """Analyze how batch size affects compression ratio and processing time."""
    print("📊 Analyzing Batch Size Effects...")
    
    pipeline = CompressionPipeline(enable_cache=False)
    dataset_text = load_dataset_file("bbc-yachts.txt")
    sample_text = dataset_text[:2000]  # Fixed size for fair comparison
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        try:
            start_time = time.time()
            compressed_data, result = pipeline.compress_with_llm_ranks_batched(
                sample_text, batch_size=batch_size
            )
            processing_time = time.time() - start_time
            
            results.append({
                'batch_size': batch_size,
                'compression_ratio': result.compression_ratio,
                'compression_percentage': result.compression_percentage,
                'processing_time': processing_time,
                'compressed_size': len(compressed_data),
                'original_size': len(sample_text.encode('utf-8'))
            })
            
            print(f"  -> {result.compression_percentage:.1f}% reduction, {processing_time:.1f}s")
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    return results

def analyze_context_window():
    """Analyze how context window size affects compression ratio and processing time."""
    print("📊 Analyzing Context Window Effects...")
    
    pipeline = CompressionPipeline(enable_cache=False)
    dataset_text = load_dataset_file("bbc-yachts.txt")
    sample_text = dataset_text[:2000]  # Fixed size for fair comparison
    
    # Test different context windows (reduced for faster execution)
    context_windows = [100, 300, 500, 1024]  # 1024 is GPT-2's max
    results = []
    
    for context_length in context_windows:
        print(f"Testing context window: {context_length}")
        
        try:
            start_time = time.time()
            compressed_data, result = pipeline.compress_with_llm_ranks_huffman_zipf_bytes(
                sample_text, max_context_length=context_length
            )
            processing_time = time.time() - start_time
            
            results.append({
                'context_length': context_length,
                'compression_ratio': result.compression_ratio,
                'compression_percentage': result.compression_percentage,
                'processing_time': processing_time,
                'compressed_size': len(compressed_data),
                'original_size': len(sample_text.encode('utf-8'))
            })
            
            print(f"  -> {result.compression_percentage:.1f}% reduction, {processing_time:.1f}s")
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    return results

def analyze_compression_methods():
    """Analyze different compression methods on the same text."""
    print("📊 Analyzing Compression Method Effects...")
    
    pipeline = CompressionPipeline(enable_cache=False)
    dataset_text = load_dataset_file("bbc-yachts.txt")
    sample_text = dataset_text[:2000]  # Fixed size for fair comparison
    
    # Test different compression methods
    methods = [
        ("Raw zlib", pipeline.compress_with_raw_zlib),
        ("Tokenizer + zlib", pipeline.compress_with_tokenizer_zlib),
        ("LLM ranks + zlib", pipeline.compress_with_llm_ranks),
        ("LLM ranks + Huffman (basic)", pipeline.compress_with_llm_ranks_huffman),
        ("LLM ranks + Huffman (Zipf)", pipeline.compress_with_llm_ranks_huffman_zipf),
        ("LLM ranks + Huffman (Zipf-bytes)", pipeline.compress_with_llm_ranks_huffman_zipf_bytes),
    ]
    
    results = []
    
    for method_name, method_func in methods:
        print(f"Testing {method_name}...")
        
        try:
            start_time = time.time()
            if method_name == "Tokenizer + zlib":
                compressed_data, result = method_func(sample_text)
            else:
                compressed_data, result = method_func(sample_text)
            processing_time = time.time() - start_time
            
            results.append({
                'method': method_name,
                'compression_ratio': result.compression_ratio,
                'compression_percentage': result.compression_percentage,
                'processing_time': processing_time,
                'compressed_size': len(compressed_data),
                'original_size': len(sample_text.encode('utf-8'))
            })
            
            print(f"  -> {result.compression_percentage:.1f}% reduction, {processing_time:.1f}s")
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    return results

def create_graphs(document_results, batch_results, context_results, method_results):
    """Create comprehensive graphs showing parameter effects."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('LLM Compression Parameter Analysis', fontsize=16, fontweight='bold')
    
    # Document Size Effects
    if document_results:
        sizes = [r['size'] for r in document_results]
        comp_ratios = [r['compression_percentage'] for r in document_results]
        times = [r['processing_time'] for r in document_results]
        
        axes[0, 0].plot(sizes, comp_ratios, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Document Size vs Compression Ratio', fontweight='bold')
        axes[0, 0].set_xlabel('Document Size (characters)')
        axes[0, 0].set_ylabel('Compression Reduction (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[1, 0].plot(sizes, times, 'r-o', linewidth=2, markersize=8)
        axes[1, 0].set_title('Document Size vs Processing Time', fontweight='bold')
        axes[1, 0].set_xlabel('Document Size (characters)')
        axes[1, 0].set_ylabel('Processing Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Batch Size Effects
    if batch_results:
        batch_sizes = [r['batch_size'] for r in batch_results]
        comp_ratios = [r['compression_percentage'] for r in batch_results]
        times = [r['processing_time'] for r in batch_results]
        
        axes[0, 1].plot(batch_sizes, comp_ratios, 'g-o', linewidth=2, markersize=8)
        axes[0, 1].set_title('Batch Size vs Compression Ratio', fontweight='bold')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Compression Reduction (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 1].plot(batch_sizes, times, 'orange', marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Batch Size vs Processing Time', fontweight='bold')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Processing Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Context Window Effects
    if context_results:
        contexts = [r['context_length'] for r in context_results]
        comp_ratios = [r['compression_percentage'] for r in context_results]
        times = [r['processing_time'] for r in context_results]
        
        axes[0, 2].plot(contexts, comp_ratios, 'purple', marker='o', linewidth=2, markersize=8)
        axes[0, 2].set_title('Context Window vs Compression Ratio', fontweight='bold')
        axes[0, 2].set_xlabel('Context Window Size (tokens)')
        axes[0, 2].set_ylabel('Compression Reduction (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 2].plot(contexts, times, 'brown', marker='o', linewidth=2, markersize=8)
        axes[1, 2].set_title('Context Window vs Processing Time', fontweight='bold')
        axes[1, 2].set_xlabel('Context Window Size (tokens)')
        axes[1, 2].set_ylabel('Processing Time (seconds)')
        axes[1, 2].grid(True, alpha=0.3)
    
    # Method Comparison
    if method_results:
        methods = [r['method'] for r in method_results]
        comp_ratios = [r['compression_percentage'] for r in method_results]
        times = [r['processing_time'] for r in method_results]
        
        # Shorten method names for display
        short_methods = [m.replace('LLM ranks + ', '').replace('Huffman ', 'H-') for m in methods]
        
        bars1 = axes[0, 3].bar(range(len(methods)), comp_ratios, color='skyblue', alpha=0.8)
        axes[0, 3].set_title('Compression Method Comparison', fontweight='bold')
        axes[0, 3].set_ylabel('Compression Reduction (%)')
        axes[0, 3].set_xticks(range(len(methods)))
        axes[0, 3].set_xticklabels(short_methods, rotation=45, ha='right')
        axes[0, 3].grid(True, alpha=0.3)
        
        bars2 = axes[1, 3].bar(range(len(methods)), times, color='lightcoral', alpha=0.8)
        axes[1, 3].set_title('Processing Time by Method', fontweight='bold')
        axes[1, 3].set_ylabel('Processing Time (seconds)')
        axes[1, 3].set_xticks(range(len(methods)))
        axes[1, 3].set_xticklabels(short_methods, rotation=45, ha='right')
        axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📈 Graphs saved to parameter_analysis.png")

def main():
    """Run comprehensive parameter analysis."""
    print("🚀 LLM Compression Parameter Analysis")
    print("="*50)
    
    # Run all analyses
    document_results = analyze_document_size()
    batch_results = analyze_batch_size()
    context_results = analyze_context_window()
    method_results = analyze_compression_methods()
    
    # Save results to JSON
    all_results = {
        'document_size': document_results,
        'batch_size': batch_results,
        'context_window': context_results,
        'compression_methods': method_results
    }
    
    with open('parameter_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create graphs
    create_graphs(document_results, batch_results, context_results, method_results)
    
    print("\n✅ Parameter analysis complete!")
    print("📊 Results saved to parameter_analysis_results.json")
    print("📈 Graphs saved to parameter_analysis.png")

if __name__ == "__main__":
    main()