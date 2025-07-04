import zlib
import pickle
import sqlite3
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from llm_ranker import LLMRanker
from ranker_cache import RankerCache


@dataclass
class CompressionResult:
    """Results from a compression experiment."""
    method: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    model_name: Optional[str] = None
    context_length: Optional[int] = None
    text_sample: str = ""  # First 100 chars for reference
    
    @property
    def compression_percentage(self) -> float:
        """Compression as percentage reduction."""
        return (1 - self.compression_ratio) * 100


class CompressionPipeline:
    """Main compression pipeline with database storage for results."""
    
    def __init__(self, db_path: str = "compression_results.db"):
        """Initialize the compression pipeline.
        
        Args:
            db_path: Path to SQLite database for storing results
        """
        self.db_path = db_path
        self.ranker: Optional[LLMRanker] = None
        self.cache = RankerCache(db_path)
        self._setup_database()
    
    def _setup_database(self):
        """Initialize the SQLite database for storing results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create compression results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compression_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                method TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                compression_time REAL NOT NULL,
                decompression_time REAL NOT NULL,
                model_name TEXT,
                context_length INTEGER,
                text_sample TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_ranker(self, model_name: str = "gpt2", max_context_length: Optional[int] = None) -> LLMRanker:
        """Get or create LLM ranker instance."""
        if (self.ranker is None or 
            self.ranker.model_name != model_name or 
            self.ranker.max_context_length != max_context_length):
            self.ranker = LLMRanker(
                model_name=model_name,
                max_context_length=max_context_length
            )
        return self.ranker
    
    def get_token_ranks_cached(self, text: str, ranker: LLMRanker) -> List[int]:
        """Get token ranks with caching."""
        return self.cache.get_token_ranks_cached(text, ranker)
    
    def get_string_from_token_ranks_cached(self, ranks: List[int], ranker: LLMRanker) -> str:
        """Get string from token ranks with caching."""
        return self.cache.get_string_from_token_ranks_cached(ranks, ranker)
    
    def compress_with_llm_ranks(
        self, 
        text: str, 
        model_name: str = "gpt2",
        max_context_length: Optional[int] = None
    ) -> Tuple[bytes, CompressionResult]:
        """Compress text using LLM rank encoding + zlib.
        
        Args:
            text: Input text to compress
            model_name: LLM model to use for ranking
            max_context_length: Maximum context length for LLM
            
        Returns:
            Tuple of (compressed_data, compression_result)
        """
        ranker = self._get_ranker(model_name, max_context_length)
        
        # Phase 1: Get token ranks (with caching)
        start_time = time.time()
        ranks = self.get_token_ranks_cached(text, ranker)
        
        # Phase 2: Compress ranks with zlib
        ranks_bytes = pickle.dumps(ranks)
        compressed_data = zlib.compress(ranks_bytes)
        compression_time = time.time() - start_time
        
        # Test decompression
        start_time = time.time()
        decompressed_ranks = pickle.loads(zlib.decompress(compressed_data))
        reconstructed_text = self.get_string_from_token_ranks_cached(decompressed_ranks, ranker)
        decompression_time = time.time() - start_time
        
        # Verify round-trip accuracy
        if reconstructed_text != text:
            raise ValueError("Round-trip compression failed - reconstructed text doesn't match original")
        
        original_size = len(text.encode('utf-8'))
        compressed_size = len(compressed_data)
        
        result = CompressionResult(
            method="llm_ranks_zlib",
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            compression_time=compression_time,
            decompression_time=decompression_time,
            model_name=model_name,
            context_length=max_context_length,
            text_sample=text[:100]
        )
        
        return compressed_data, result
    
    def compress_with_raw_zlib(self, text: str) -> Tuple[bytes, CompressionResult]:
        """Baseline: compress raw text with zlib.
        
        Args:
            text: Input text to compress
            
        Returns:
            Tuple of (compressed_data, compression_result)
        """
        text_bytes = text.encode('utf-8')
        
        # Compression
        start_time = time.time()
        compressed_data = zlib.compress(text_bytes)
        compression_time = time.time() - start_time
        
        # Test decompression
        start_time = time.time()
        decompressed_data = zlib.decompress(compressed_data)
        decompression_time = time.time() - start_time
        
        # Verify round-trip
        if decompressed_data.decode('utf-8') != text:
            raise ValueError("Round-trip compression failed")
        
        original_size = len(text_bytes)
        compressed_size = len(compressed_data)
        
        result = CompressionResult(
            method="raw_zlib",
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            compression_time=compression_time,
            decompression_time=decompression_time,
            text_sample=text[:100]
        )
        
        return compressed_data, result
    
    def compress_with_tokenizer_zlib(
        self, 
        text: str, 
        model_name: str = "gpt2"
    ) -> Tuple[bytes, CompressionResult]:
        """Baseline: tokenize then compress token IDs with zlib.
        
        Args:
            text: Input text to compress
            model_name: Tokenizer model to use
            
        Returns:
            Tuple of (compressed_data, compression_result)
        """
        ranker = self._get_ranker(model_name)
        
        # Tokenize
        start_time = time.time()
        token_ids = ranker.tokenizer.encode(text)
        token_bytes = pickle.dumps(token_ids)
        compressed_data = zlib.compress(token_bytes)
        compression_time = time.time() - start_time
        
        # Test decompression
        start_time = time.time()
        decompressed_tokens = pickle.loads(zlib.decompress(compressed_data))
        reconstructed_text = ranker.tokenizer.decode(decompressed_tokens, skip_special_tokens=True)
        decompression_time = time.time() - start_time
        
        original_size = len(text.encode('utf-8'))
        compressed_size = len(compressed_data)
        
        result = CompressionResult(
            method="tokenizer_zlib",
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            compression_time=compression_time,
            decompression_time=decompression_time,
            model_name=model_name,
            text_sample=text[:100]
        )
        
        return compressed_data, result
    
    def run_compression_benchmark(
        self, 
        text: str, 
        model_name: str = "gpt2",
        max_context_length: Optional[int] = None,
        save_to_db: bool = True
    ) -> Dict[str, CompressionResult]:
        """Run all compression methods on text and return results.
        
        Args:
            text: Input text to compress
            model_name: LLM model to use
            max_context_length: Maximum context length for LLM
            save_to_db: Whether to save results to database
            
        Returns:
            Dictionary mapping method names to results
        """
        results = {}
        
        # Run all compression methods
        try:
            _, results["llm_ranks_zlib"] = self.compress_with_llm_ranks(
                text, model_name, max_context_length
            )
        except Exception as e:
            print(f"LLM ranks compression failed: {e}")
            results["llm_ranks_zlib"] = None
        
        try:
            _, results["raw_zlib"] = self.compress_with_raw_zlib(text)
        except Exception as e:
            print(f"Raw zlib compression failed: {e}")
            results["raw_zlib"] = None
        
        try:
            _, results["tokenizer_zlib"] = self.compress_with_tokenizer_zlib(text, model_name)
        except Exception as e:
            print(f"Tokenizer zlib compression failed: {e}")
            results["tokenizer_zlib"] = None
        
        # Save to database if requested
        if save_to_db:
            for method, result in results.items():
                if result is not None:
                    self._save_result_to_db(result)
        
        return results
    
    def _save_result_to_db(self, result: CompressionResult):
        """Save a compression result to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO compression_results (
                timestamp, method, original_size, compressed_size, compression_ratio,
                compression_time, decompression_time, model_name, context_length, 
                text_sample, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.strftime('%Y-%m-%d %H:%M:%S'),
            result.method,
            result.original_size,
            result.compressed_size,
            result.compression_ratio,
            result.compression_time,
            result.decompression_time,
            result.model_name,
            result.context_length,
            result.text_sample,
            json.dumps(asdict(result))
        ))
        
        conn.commit()
        conn.close()
    
    def get_results_from_db(self, method: Optional[str] = None) -> List[Dict]:
        """Retrieve compression results from database.
        
        Args:
            method: Filter by compression method (optional)
            
        Returns:
            List of result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if method:
            cursor.execute("""
                SELECT * FROM compression_results WHERE method = ?
                ORDER BY timestamp DESC
            """, (method,))
        else:
            cursor.execute("""
                SELECT * FROM compression_results ORDER BY timestamp DESC
            """)
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def print_comparison_table(self, results: Dict[str, CompressionResult]):
        """Print a formatted comparison table of compression results."""
        if not any(results.values()):
            print("No successful compression results to display.")
            return
        
        print("\n" + "="*80)
        print("COMPRESSION COMPARISON RESULTS")
        print("="*80)
        
        # Table header
        print(f"{'Method':<20} {'Original':<10} {'Compressed':<10} {'Ratio':<8} {'Reduction':<10} {'Time (s)':<10}")
        print("-" * 80)
        
        # Sort by compression ratio (best first)
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if v is not None],
            key=lambda x: x[1].compression_ratio
        )
        
        for method, result in sorted_results:
            print(f"{method:<20} {result.original_size:<10} {result.compressed_size:<10} "
                  f"{result.compression_ratio:<8.3f} {result.compression_percentage:<9.1f}% "
                  f"{result.compression_time:<10.3f}")
        
        print("-" * 80)
        
        # Show best performing method
        if sorted_results:
            best_method, best_result = sorted_results[0]
            print(f"Best compression: {best_method} ({best_result.compression_percentage:.1f}% reduction)")
        
        print("="*80)


def load_dataset_file(filename: str) -> str:
    """Load a text file from the datasets directory."""
    file_path = Path("datasets") / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    """Example usage of the compression pipeline."""
    pipeline = CompressionPipeline()
    
    # Test with a simple example
    test_text = "Hello world! This is a test of the compression pipeline. " * 10
    print(f"Testing with sample text ({len(test_text)} characters)")
    
    results = pipeline.run_compression_benchmark(test_text)
    pipeline.print_comparison_table(results)


if __name__ == "__main__":
    main()