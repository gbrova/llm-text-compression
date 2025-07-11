import pytest
import torch
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
    
    def test_batched_perfect_recovery(self):
        """Test that batching still achieves perfect recovery."""
        text = "The quick brown fox jumps over the lazy dog"
        
        # Test with different batch sizes
        for batch_size in [1, 2, 3, 5]:
            ranker = LLMRanker(batch_size=batch_size)
            ranks = ranker.get_token_ranks(text)
            reconstructed = ranker.get_string_from_token_ranks(ranks)
            
            # Should achieve perfect recovery regardless of batch size
            assert reconstructed == text, f"Perfect recovery failed with batch_size={batch_size}"
    
    def test_batch_size_consistency(self):
        """Test that batching produces consistent results within each batch."""
        text = "The quick brown fox jumps over the lazy dog"
        
        # When processing in batches, each batch should be internally consistent
        # but different batch sizes will produce different results due to context reset
        ranker = LLMRanker(batch_size=2)
        ranks = ranker.get_token_ranks(text)
        
        # Should produce valid ranks
        assert isinstance(ranks, list)
        assert len(ranks) > 0
        assert all(isinstance(rank, int) for rank in ranks)
        assert all(rank >= 1 for rank in ranks)
    
    def test_batching_with_uneven_division(self):
        """Test that batching works correctly when tokens don't divide evenly."""
        # Use a text that generates 10 tokens when n=3 (like the example in the requirements)
        text = "one two three four five six seven eight nine ten"
        
        ranker = LLMRanker(batch_size=3)
        ranks = ranker.get_token_ranks(text)
        reconstructed = ranker.get_string_from_token_ranks(ranks)
        
        # Should achieve perfect recovery even with uneven division
        assert reconstructed == text
    
    def test_token_chunk_creation_logic(self):
        """Test the token chunk creation logic with specific scenarios."""
        ranker = LLMRanker(batch_size=3)
        
        # Test with 10 tokens (should create chunks of size 4, 3, 3)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        chunks = ranker._create_chunks(tokens, 3)
        
        assert len(chunks) == 3
        assert chunks[0].shape[1] == 4  # First chunk gets 4 tokens
        assert chunks[1].shape[1] == 3  # Second chunk gets 3 tokens
        assert chunks[2].shape[1] == 3  # Third chunk gets 3 tokens
        
        # Verify all tokens are preserved
        all_tokens = torch.cat(chunks, dim=1)
        assert torch.equal(all_tokens, tokens)
    
    def test_rank_chunk_creation_logic(self):
        """Test the rank chunk creation logic with specific scenarios."""
        ranker = LLMRanker(batch_size=3)
        
        # Test with 10 ranks (should create chunks of size 4, 3, 3)
        ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ranks_tensor = torch.tensor([ranks], dtype=torch.long)  # Convert to tensor format
        chunks = ranker._create_chunks(ranks_tensor, 3)
        
        assert len(chunks) == 3
        assert chunks[0].shape[1] == 4  # First chunk gets 4 ranks
        assert chunks[1].shape[1] == 3  # Second chunk gets 3 ranks
        assert chunks[2].shape[1] == 3  # Third chunk gets 3 ranks
        
        # Verify all ranks are preserved
        all_chunks_tensor = torch.cat(chunks, dim=1)
        assert torch.equal(all_chunks_tensor, ranks_tensor)
    
    def test_empty_input_batched(self):
        """Test empty input with batching."""
        ranker = LLMRanker(batch_size=3)
        
        # Test empty text
        ranks = ranker.get_token_ranks("")
        assert ranks == []
        
        # Test empty ranks
        result = ranker.get_string_from_token_ranks([])
        assert result == ""
    
    def test_single_token_batched(self):
        """Test single token with batching."""
        ranker = LLMRanker(batch_size=3)
        
        text = "Hello"
        ranks = ranker.get_token_ranks(text)
        reconstructed = ranker.get_string_from_token_ranks(ranks)
        
        assert reconstructed == text


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__, "-v"])