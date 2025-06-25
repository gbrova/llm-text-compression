import pytest
from llm_ranker import LLMRanker, rank_tokens


class TestLLMRanker:
    @pytest.fixture
    def ranker(self):
        """Create a ranker instance for testing."""
        return LLMRanker()
    
    def test_convenience_function(self):
        """Test the rank_tokens convenience function."""
        text = "Hello world"
        ranks = rank_tokens(text)
        
        # Should return a list of integers
        assert isinstance(ranks, list)
        assert len(ranks) == 2  # "Hello" and "world" tokens
        assert all(isinstance(rank, int) for rank in ranks)
        assert all(rank >= 1 for rank in ranks)  # Ranks are 1-indexed
    
    def test_ranker_properties(self, ranker):
        """Test basic ranker properties."""
        vocab_size = ranker.get_vocab_size()
        context_length = ranker.get_context_length()
        
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
        assert isinstance(context_length, int)
        assert context_length > 0
        
        # GPT-2 has 50,257 tokens and 1024 context length by default
        assert vocab_size == 50257
        assert context_length == 1024
    
    def test_get_token_ranks_simple(self, ranker):
        """Test token ranking with simple text."""
        text = "Hello world"
        ranks = ranker.get_token_ranks(text)
        
        assert isinstance(ranks, list)
        assert len(ranks) == 2
        assert all(isinstance(rank, int) for rank in ranks)
        assert all(rank >= 1 for rank in ranks)
        assert all(rank <= ranker.get_vocab_size() for rank in ranks)
    
    def test_get_token_ranks_longer(self, ranker):
        """Test token ranking with longer text."""
        text = "The quick brown fox jumps over the lazy dog"
        ranks = ranker.get_token_ranks(text)
        
        assert isinstance(ranks, list)
        assert len(ranks) > 5  # Should have multiple tokens
        assert all(isinstance(rank, int) for rank in ranks)
        assert all(rank >= 1 for rank in ranks)
        assert all(rank <= ranker.get_vocab_size() for rank in ranks)
    
    def test_get_token_ranks_empty(self, ranker):
        """Test token ranking with empty text."""
        ranks = ranker.get_token_ranks("")
        assert ranks == []
    
    def test_round_trip_reconstruction(self, ranker):
        """Test that we can reconstruct text from ranks."""
        original_text = "Hello world"
        ranks = ranker.get_token_ranks(original_text)
        reconstructed = ranker.get_string_from_token_ranks(ranks)
        
        # Should be able to reconstruct the original text
        assert reconstructed == original_text
    
    def test_get_string_from_token_ranks_custom(self, ranker):
        """Test string generation with custom ranks."""
        # Test with rank 1 (most likely tokens)
        custom_ranks = [1, 1, 1]
        result = ranker.get_string_from_token_ranks(custom_ranks)
        
        assert isinstance(result, str)
        assert len(result) > 0  # Should generate some text
    
    def test_get_string_from_token_ranks_empty(self, ranker):
        """Test string generation with empty ranks."""
        result = ranker.get_string_from_token_ranks([])
        assert result == ""
    
    def test_get_string_from_token_ranks_out_of_bounds(self, ranker):
        """Test string generation with out-of-bounds ranks."""
        # Use a rank higher than vocab size
        very_high_rank = ranker.get_vocab_size() + 1000
        ranks = [very_high_rank]
        result = ranker.get_string_from_token_ranks(ranks)
        
        # Should handle gracefully and return some text
        assert isinstance(result, str)
    
    def test_consistent_results(self, ranker):
        """Test that the same input produces consistent results."""
        text = "Hello world"
        
        ranks1 = ranker.get_token_ranks(text)
        ranks2 = ranker.get_token_ranks(text)
        
        # Results should be deterministic
        assert ranks1 == ranks2


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__, "-v"])