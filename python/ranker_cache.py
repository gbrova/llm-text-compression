import sqlite3
import json
import hashlib
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_ranker import LLMRanker


class RankerCache:
    """
    Cache for LLM ranker operations to avoid recomputing expensive token ranking operations.
    
    Stores cached results for:
    - get_token_ranks: text -> ranks
    - get_string_from_token_ranks: ranks -> text
    
    Cache keys are based on ranker signature to ensure different ranker configurations
    don't interfere with each other.
    """
    
    def __init__(self, db_path: str = "compression_results.db"):
        """
        Initialize the ranker cache.
        
        Args:
            db_path: Path to SQLite database for caching
        """
        self.db_path = db_path
        self._setup_cache_tables()
    
    def _setup_cache_tables(self):
        """Initialize the cache tables in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create token ranks cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_ranks_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                ranker_signature TEXT NOT NULL,
                text_sample TEXT NOT NULL,
                ranks TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(text_hash, ranker_signature)
            )
        """)
        
        # Create string from ranks cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS string_from_ranks_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ranks_hash TEXT NOT NULL,
                ranker_signature TEXT NOT NULL,
                ranks_sample TEXT NOT NULL,
                result_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ranks_hash, ranker_signature)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for the input text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_ranks_hash(self, ranks: List[int]) -> str:
        """Generate a hash for the input ranks."""
        ranks_str = json.dumps(ranks)
        return hashlib.sha256(ranks_str.encode('utf-8')).hexdigest()
    
    def _get_ranker_signature_str(self, ranker: 'LLMRanker') -> str:
        """Get a string representation of the ranker signature for database storage."""
        signature = ranker.get_cache_signature()
        return json.dumps(signature, sort_keys=True)
    
    def get_cached_token_ranks(self, text: str, ranker: 'LLMRanker') -> Optional[List[int]]:
        """
        Retrieve cached token ranks if available.
        
        Args:
            text: Input text
            ranker: LLM ranker instance
            
        Returns:
            Cached ranks if found, None otherwise
        """
        text_hash = self._get_text_hash(text)
        ranker_signature = self._get_ranker_signature_str(ranker)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ranks FROM token_ranks_cache 
            WHERE text_hash = ? AND ranker_signature = ?
        """, (text_hash, ranker_signature))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def cache_token_ranks(self, text: str, ranker: 'LLMRanker', ranks: List[int]):
        """
        Cache the computed token ranks.
        
        Args:
            text: Input text
            ranker: LLM ranker instance
            ranks: Computed token ranks
        """
        text_hash = self._get_text_hash(text)
        ranker_signature = self._get_ranker_signature_str(ranker)
        ranks_json = json.dumps(ranks)
        # Store first 100 characters of text as sample for debugging
        text_sample = text[:100]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO token_ranks_cache 
            (text_hash, ranker_signature, text_sample, ranks)
            VALUES (?, ?, ?, ?)
        """, (text_hash, ranker_signature, text_sample, ranks_json))
        
        conn.commit()
        conn.close()
    
    def get_cached_string_from_ranks(self, ranks: List[int], ranker: 'LLMRanker') -> Optional[str]:
        """
        Retrieve cached string from ranks if available.
        
        Args:
            ranks: Input ranks
            ranker: LLM ranker instance
            
        Returns:
            Cached string if found, None otherwise
        """
        ranks_hash = self._get_ranks_hash(ranks)
        ranker_signature = self._get_ranker_signature_str(ranker)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT result_text FROM string_from_ranks_cache 
            WHERE ranks_hash = ? AND ranker_signature = ?
        """, (ranks_hash, ranker_signature))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        return None
    
    def cache_string_from_ranks(self, ranks: List[int], ranker: 'LLMRanker', result_text: str):
        """
        Cache the computed string from ranks.
        
        Args:
            ranks: Input ranks
            ranker: LLM ranker instance
            result_text: Computed result text
        """
        ranks_hash = self._get_ranks_hash(ranks)
        ranker_signature = self._get_ranker_signature_str(ranker)
        ranks_json = json.dumps(ranks)
        # Store first 100 characters of ranks as sample for debugging
        ranks_sample = ranks_json[:100]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO string_from_ranks_cache 
            (ranks_hash, ranker_signature, ranks_sample, result_text)
            VALUES (?, ?, ?, ?)
        """, (ranks_hash, ranker_signature, ranks_sample, result_text))
        
        conn.commit()
        conn.close()
    
    def get_token_ranks_cached(self, text: str, ranker: 'LLMRanker') -> List[int]:
        """
        Get token ranks with caching.
        
        Args:
            text: Input text
            ranker: LLM ranker instance
            
        Returns:
            Token ranks (from cache or computed)
        """
        # Check cache first
        cached_ranks = self.get_cached_token_ranks(text, ranker)
        if cached_ranks is not None:
            return cached_ranks
        
        # Not in cache, compute using ranker
        ranks = ranker.get_token_ranks(text)
        
        # Cache the result
        self.cache_token_ranks(text, ranker, ranks)
        
        return ranks
    
    def get_string_from_token_ranks_cached(self, ranks: List[int], ranker: 'LLMRanker') -> str:
        """
        Get string from token ranks with caching.
        
        Args:
            ranks: Input ranks
            ranker: LLM ranker instance
            
        Returns:
            Generated text (from cache or computed)
        """
        # Check cache first
        cached_result = self.get_cached_string_from_ranks(ranks, ranker)
        if cached_result is not None:
            return cached_result
        
        # Not in cache, compute using ranker
        result_text = ranker.get_string_from_token_ranks(ranks)
        
        # Cache the result
        self.cache_string_from_ranks(ranks, ranker, result_text)
        
        return result_text
    
    def clear_cache(self):
        """Clear all cached data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM token_ranks_cache")
        cursor.execute("DELETE FROM string_from_ranks_cache")
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self) -> dict:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get token ranks cache stats
        cursor.execute("SELECT COUNT(*) FROM token_ranks_cache")
        token_ranks_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM string_from_ranks_cache")
        string_ranks_count = cursor.fetchone()[0]
        
        # Get unique ranker signatures
        cursor.execute("SELECT DISTINCT ranker_signature FROM token_ranks_cache")
        token_signatures = cursor.fetchall()
        
        cursor.execute("SELECT DISTINCT ranker_signature FROM string_from_ranks_cache")
        string_signatures = cursor.fetchall()
        
        conn.close()
        
        return {
            "token_ranks_entries": token_ranks_count,
            "string_from_ranks_entries": string_ranks_count,
            "total_entries": token_ranks_count + string_ranks_count,
            "unique_ranker_signatures_token": len(token_signatures),
            "unique_ranker_signatures_string": len(string_signatures)
        }