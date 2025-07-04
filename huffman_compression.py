"""
Huffman compression utilities for LLM rank sequences.

This module provides Huffman coding compression methods for token rank sequences,
including both basic frequency table approach and parametric Zipf distribution approach.
"""

import pickle
import time
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from scipy.optimize import minimize_scalar
from dahuffman import HuffmanCodec


class HuffmanRankCompressor:
    """Huffman compression for LLM token rank sequences."""
    
    def compress_basic(self, ranks: List[int]) -> bytes:
        """Compress ranks using basic Huffman coding with frequency table.
        
        Args:
            ranks: List of token ranks
            
        Returns:
            Compressed data as bytes
        """
        # Build Huffman codec from rank frequencies
        rank_counts = Counter(ranks)
        codec = HuffmanCodec.from_frequencies(rank_counts)
        
        # Compress ranks
        compressed_ranks = codec.encode(ranks)
        
        # Store both the codec and compressed data
        compressed_data = pickle.dumps({
            'codec': pickle.dumps(codec),
            'compressed_ranks': compressed_ranks
        })
        
        return compressed_data
    
    def decompress_basic(self, compressed_data: bytes) -> List[int]:
        """Decompress ranks from basic Huffman compressed data.
        
        Args:
            compressed_data: Compressed data as bytes
            
        Returns:
            List of decompressed token ranks
        """
        # Load compressed data
        data = pickle.loads(compressed_data)
        codec = pickle.loads(data['codec'])
        
        # Decompress ranks
        decompressed_ranks = codec.decode(data['compressed_ranks'])
        return decompressed_ranks
    
    def _fit_zipf_distribution(self, rank_counts: Dict[int, int]) -> Tuple[float, int]:
        """Fit a Zipf distribution to rank frequencies.
        
        Args:
            rank_counts: Dictionary mapping ranks to their frequencies
            
        Returns:
            Tuple of (zipf_s_parameter, max_rank)
        """
        # Convert to arrays for fitting
        ranks = np.array(list(rank_counts.keys()))
        counts = np.array(list(rank_counts.values()))
        
        # Fit Zipf distribution: f(k) = k^(-s) / sum(i^(-s))
        # We'll find the s parameter that minimizes the sum of squared errors
        def zipf_error(s):
            if s <= 0:
                return float('inf')
            # Predicted counts based on Zipf distribution
            total_count = sum(counts)
            normalization = sum(1.0 / (i**s) for i in ranks)
            predicted = total_count * (1.0 / (ranks**s)) / normalization
            return np.sum((counts - predicted)**2)
        
        # Find optimal s parameter
        result = minimize_scalar(zipf_error, bounds=(0.1, 5.0), method='bounded')
        s_param = result.x
        max_rank = max(ranks)
        
        return s_param, max_rank
    
    def _generate_zipf_frequencies(self, s_param: float, max_rank: int, total_count: int) -> Dict[int, int]:
        """Generate frequencies based on Zipf distribution parameters.
        
        Args:
            s_param: Zipf distribution parameter
            max_rank: Maximum rank value
            total_count: Total number of tokens
            
        Returns:
            Dictionary mapping ranks to their predicted frequencies
        """
        frequencies = {}
        normalization = sum(1.0 / (i**s_param) for i in range(1, max_rank + 1))
        
        for rank in range(1, max_rank + 1):
            prob = (1.0 / (rank**s_param)) / normalization
            freq = max(1, int(total_count * prob))  # Ensure at least 1
            frequencies[rank] = freq
        
        return frequencies
    
    def compress_zipf(self, ranks: List[int]) -> bytes:
        """Compress ranks using Zipf distribution parametric Huffman coding.
        
        Args:
            ranks: List of token ranks
            
        Returns:
            Compressed data as bytes
        """
        # Fit Zipf distribution and create parametric Huffman codec
        rank_counts = Counter(ranks)
        s_param, max_rank = self._fit_zipf_distribution(rank_counts)
        
        # Generate frequencies based on fitted Zipf distribution
        zipf_frequencies = self._generate_zipf_frequencies(s_param, max_rank, len(ranks))
        
        # Build Huffman codec from parametric distribution
        codec = HuffmanCodec.from_frequencies(zipf_frequencies)
        
        # Compress ranks
        compressed_ranks = codec.encode(ranks)
        
        # Store only the distribution parameters and compressed data
        compressed_data = pickle.dumps({
            'zipf_s': s_param,
            'max_rank': max_rank,
            'total_count': len(ranks),
            'compressed_ranks': compressed_ranks
        })
        
        return compressed_data
    
    def decompress_zipf(self, compressed_data: bytes) -> List[int]:
        """Decompress ranks from Zipf parametric Huffman compressed data.
        
        Args:
            compressed_data: Compressed data as bytes
            
        Returns:
            List of decompressed token ranks
        """
        # Load compressed data
        data = pickle.loads(compressed_data)
        
        # Reconstruct codec from parameters
        zipf_frequencies = self._generate_zipf_frequencies(
            data['zipf_s'], 
            data['max_rank'], 
            data['total_count']
        )
        codec = HuffmanCodec.from_frequencies(zipf_frequencies)
        
        # Decompress ranks
        decompressed_ranks = codec.decode(data['compressed_ranks'])
        return decompressed_ranks
    
    def compress_ranks_with_timing(self, ranks: List[int], method: str = 'basic') -> Tuple[bytes, float]:
        """Compress ranks with timing information.
        
        Args:
            ranks: List of token ranks
            method: Compression method ('basic' or 'zipf')
            
        Returns:
            Tuple of (compressed_data, compression_time)
        """
        start_time = time.time()
        
        if method == 'basic':
            compressed_data = self.compress_basic(ranks)
        elif method == 'zipf':
            compressed_data = self.compress_zipf(ranks)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        compression_time = time.time() - start_time
        return compressed_data, compression_time
    
    def decompress_ranks_with_timing(self, compressed_data: bytes, method: str = 'basic') -> Tuple[List[int], float]:
        """Decompress ranks with timing information.
        
        Args:
            compressed_data: Compressed data as bytes
            method: Compression method ('basic' or 'zipf')
            
        Returns:
            Tuple of (decompressed_ranks, decompression_time)
        """
        start_time = time.time()
        
        if method == 'basic':
            decompressed_ranks = self.decompress_basic(compressed_data)
        elif method == 'zipf':
            decompressed_ranks = self.decompress_zipf(compressed_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        decompression_time = time.time() - start_time
        return decompressed_ranks, decompression_time