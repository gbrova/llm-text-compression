#!/usr/bin/env python3
"""
Test script for the rank visualizer functionality.
"""

from rank_visualizer import get_tokens_and_ranks
import json

def test_basic_functionality():
    """Test basic token ranking and visualization."""
    print("🧪 Testing rank visualizer functionality...")
    
    # Test with simple text
    text = "Hello world"
    print(f"\n📝 Input text: '{text}'")
    
    try:
        tokens_with_ranks = get_tokens_and_ranks(text)
        print(f"✅ Successfully analyzed {len(tokens_with_ranks)} tokens")
        
        for i, token_data in enumerate(tokens_with_ranks):
            print(f"  Token {i+1}: '{token_data['token']}' (rank: {token_data['rank']}, color: {token_data['color']})")
        
        # Test with longer text
        longer_text = "The quick brown fox jumps over the lazy dog."
        print(f"\n📝 Input text: '{longer_text}'")
        
        tokens_with_ranks = get_tokens_and_ranks(longer_text)
        print(f"✅ Successfully analyzed {len(tokens_with_ranks)} tokens")
        
        # Show first few tokens
        for i, token_data in enumerate(tokens_with_ranks[:5]):
            print(f"  Token {i+1}: '{token_data['token']}' (rank: {token_data['rank']}, color: {token_data['color']})")
        
        if len(tokens_with_ranks) > 5:
            print(f"  ... and {len(tokens_with_ranks) - 5} more tokens")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_basic_functionality()