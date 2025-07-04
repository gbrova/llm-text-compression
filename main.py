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
    
    # Test 1: Simple repeated text (should compress very well)
    print("\n1. Testing with repeated text pattern:")
    repeated_text = "The quick brown fox jumps over the lazy dog. " * 20
    results = pipeline.run_compression_benchmark(repeated_text)
    pipeline.print_comparison_table(results)
    
    # Test 2: Load and test a real dataset if available
    if dataset_files:
        print(f"\n2. Testing with real dataset: {dataset_files[0].name}")
        try:
            dataset_text = load_dataset_file(dataset_files[0].name)
            # Use first 1000 characters for faster testing
            sample_text = dataset_text[:1000]
            print(f"   Using first 1000 characters ({len(sample_text)} chars)")
            
            results = pipeline.run_compression_benchmark(sample_text)
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
