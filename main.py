from compression_pipeline import CompressionPipeline, load_dataset_file
from pathlib import Path


def main():
    """Main entry point for LLM compression experiments."""
    print("🚀 LLM Compression Pipeline - Phase 2")
    print("="*50)
    
    pipeline = CompressionPipeline()
    
    # List available datasets
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        dataset_files = list(datasets_dir.glob("*.txt"))
        print(f"\nAvailable datasets: {len(dataset_files)} files")
        for file in dataset_files:
            print(f"  - {file.name}")
    
    # Run benchmarks on different text samples
    print("\n📊 Running compression benchmarks...")
    
    # Test 1: Load and test bbc-yachts.txt with 3k characters
    if dataset_files:
        print(f"\n1. Testing with real dataset: bbc-yachts.txt (3k characters)")
        try:
            dataset_text = load_dataset_file("bbc-yachts.txt")
            # Use first 3000 characters
            sample_text = dataset_text[:3000]
            print(f"   Using first 3000 characters ({len(sample_text)} chars)")
            
            results = pipeline.run_compression_benchmark(sample_text)
            pipeline.print_comparison_table(results)
        except Exception as e:
            print(f"   Error loading dataset: {e}")
            
        # Test 2: Same dataset with max context length of 300
        print(f"\n2. Testing with bbc-yachts.txt (3k characters, max context length 300)")
        try:
            dataset_text = load_dataset_file("bbc-yachts.txt")
            sample_text = dataset_text[:3000]
            print(f"   Using first 3000 characters ({len(sample_text)} chars), max context length 300")
            
            results = pipeline.run_compression_benchmark(sample_text, max_context_length=300)
            pipeline.print_comparison_table(results)
        except Exception as e:
            print(f"   Error loading dataset: {e}")
    
    # Test 3: Show database summary
    print("\n📈 Database Summary:")
    all_results = pipeline.get_results_from_db()
    if all_results:
        print(f"   Total experiments stored: {len(all_results)}")
        
        # Group by method
        methods = {}
        for result in all_results:
            method = result['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result['compression_ratio'])
        
        print("\n   Average compression ratios by method:")
        for method, ratios in methods.items():
            avg_ratio = sum(ratios) / len(ratios)
            avg_reduction = (1 - avg_ratio) * 100
            print(f"     {method}: {avg_ratio:.3f} ({avg_reduction:.1f}% reduction, {len(ratios)} samples)")
    else:
        print("   No experiments stored yet.")
    
    print(f"\n💾 Results saved to: {pipeline.db_path}")
    print("✅ Phase 2 compression pipeline complete!")


if __name__ == "__main__":
    main()
