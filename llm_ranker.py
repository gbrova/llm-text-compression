import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Optional


class LLMRanker:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        max_context_length: Optional[int] = None
    ):
        """
        Initialize the LLM ranker with a specified model.
        
        Args:
            model_name: HuggingFace model name (default: "gpt2")
            device: Device to run model on (default: auto-detect)
            max_context_length: Maximum context length (default: model's max)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set context length
        self.max_context_length = max_context_length or self.model.config.n_positions
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _truncate_kv_cache(self, past_key_values):
        """
        Truncate KV cache to respect max context length.
        
        Args:
            past_key_values: The past key values from the model
            
        Returns:
            Truncated past_key_values or None if input is None
        """
        if past_key_values is None or len(past_key_values) == 0:
            return past_key_values
        
        # Each element in past_key_values is a tuple of (key, value) tensors
        # The key/value tensors have shape [batch_size, num_heads, seq_len, head_dim]
        past_seq_len = past_key_values[0][0].shape[2]
        
        if past_seq_len >= self.max_context_length:
            # Truncate past_key_values to respect max context length
            truncated_past_key_values = []
            for layer_past in past_key_values:
                key, value = layer_past
                # Keep only the last (max_context_length - 1) tokens
                truncated_key = key[:, :, -(self.max_context_length - 1):, :]
                truncated_value = value[:, :, -(self.max_context_length - 1):, :]
                truncated_past_key_values.append((truncated_key, truncated_value))
            return tuple(truncated_past_key_values)
        
        return past_key_values
    
    def get_token_ranks(self, text: str) -> List[int]:
        """
        Get the rank of each token in the sequence given previous context.
        Uses KV caching for efficient computation.
        
        Args:
            text: Input text sequence
            
        Returns:
            List of ranks for each token (1-indexed, where 1 is most likely)
        """
        # Tokenize the input
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        if tokens.shape[1] == 0:
            return []
        
        ranks = []
        past_key_values = None
        
        with torch.no_grad():
            for i in range(tokens.shape[1]):  # Start from 0 to include first token
                if i == 0:
                    # For first token, use BOS token as context
                    bos_context = torch.tensor([[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]], dtype=torch.long).to(self.device)
                    outputs = self.model(bos_context, use_cache=True)
                    logits = outputs.logits[0, -1, :]
                    past_key_values = outputs.past_key_values
                else:
                    # Use single new token with cached past_key_values
                    new_token = tokens[:, i-1:i]  # Previous token (what we're predicting next from)
                    
                    # Truncate KV cache if needed to respect context length limits
                    past_key_values = self._truncate_kv_cache(past_key_values)
                    
                    outputs = self.model(new_token, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[0, -1, :]
                    past_key_values = outputs.past_key_values
                
                # Get probabilities and sort by likelihood
                probs = torch.softmax(logits, dim=-1)
                sorted_indices = torch.argsort(probs, descending=True)
                
                # Find rank of actual token
                actual_token = tokens[0, i].item()
                rank = (sorted_indices == actual_token).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
        
        return ranks
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the model."""
        return self.model.config.vocab_size
    
    def get_context_length(self) -> int:
        """Get the maximum context length being used."""
        return self.max_context_length
    
    def get_cache_signature(self) -> tuple:
        """
        Get a signature tuple that uniquely identifies this ranker configuration.
        
        This signature should include all parameters that affect the ranker's behavior
        and would make caching results from different configurations invalid.
        
        Returns:
            Tuple containing all parameters that affect ranker behavior
        """
        return (
            self.model_name,
            self.max_context_length,
            self.device  # Include device as it might affect numerical precision
        )
    
    def get_string_from_token_ranks(self, ranks: List[int], max_length: Optional[int] = None) -> str:
        """
        Generate a string by selecting tokens based on their ranks.
        Uses KV caching for efficient computation.
        
        Args:
            ranks: List of ranks (1-indexed) for each position
            max_length: Maximum length to generate (default: len(ranks))
            
        Returns:
            Generated text string
        """
        if not ranks:
            return ""
        
        max_length = max_length or len(ranks)
        tokens = []
        past_key_values = None
        
        with torch.no_grad():
            for i in range(min(len(ranks), max_length)):
                if i == 0:
                    # For first token, use BOS token as context
                    bos_context = torch.tensor([[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]], dtype=torch.long).to(self.device)
                    outputs = self.model(bos_context, use_cache=True)
                    logits = outputs.logits[0, -1, :]
                    past_key_values = outputs.past_key_values
                else:
                    # Use single new token with cached past_key_values
                    new_token = torch.tensor([[tokens[i-1]]], dtype=torch.long).to(self.device)
                    
                    # Truncate KV cache if needed to respect context length limits
                    past_key_values = self._truncate_kv_cache(past_key_values)
                    
                    outputs = self.model(new_token, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[0, -1, :]
                    past_key_values = outputs.past_key_values
                
                # Get probabilities and sort by likelihood
                probs = torch.softmax(logits, dim=-1)
                sorted_indices = torch.argsort(probs, descending=True)
                
                # Select token at the specified rank (convert from 1-indexed to 0-indexed)
                target_rank = ranks[i] - 1
                if target_rank < len(sorted_indices):
                    selected_token = sorted_indices[target_rank].item()
                else:
                    # If rank is out of bounds, use the least likely token
                    selected_token = sorted_indices[-1].item()
                
                tokens.append(selected_token)
        
        # Decode tokens to string
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def rank_tokens(text: str, model_name: str = "gpt2") -> List[int]:
    """
    Convenience function to get token ranks for a text string.
    
    Args:
        text: Input text sequence
        model_name: HuggingFace model name
        
    Returns:
        List of ranks for each token
    """
    ranker = LLMRanker(model_name=model_name)
    return ranker.get_token_ranks(text)