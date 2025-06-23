from llm_ranker import LLMRanker, rank_tokens

# Test the module
if __name__ == "__main__":
    # Simple test
    text = "Hello world"
    print(f"Input: '{text}'")
    
    # Test with convenience function
    ranks = rank_tokens(text)
    print(f"Token ranks: {ranks}")
    
    # Test with class interface
    ranker = LLMRanker()
    print(f"Vocab size: {ranker.get_vocab_size()}")
    print(f"Context length: {ranker.get_context_length()}")
    
    # Test longer sequence
    longer_text = "The quick brown fox jumps over the lazy dog"
    longer_ranks = ranker.get_token_ranks(longer_text)
    print(f"Longer text: '{longer_text}'")
    print(f"Ranks: {longer_ranks}")