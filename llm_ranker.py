import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Optional


class LLMRanker:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        max_context_length: Optional[int] = None,
        batch_size: int = 1
    ):
        """
        Initialize the LLM ranker with a specified model.
        
        Args:
            model_name: HuggingFace model name (default: "gpt2")
            device: Device to run model on (default: auto-detect)
            max_context_length: Maximum context length (default: model's max)
            batch_size: Number of parallel batches to process (default: 1)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        
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
        Uses KV caching for efficient computation and optional batch processing.
        
        When batch_size > 1, splits input into chunks and processes multiple
        chunks simultaneously using the model's batch dimension for improved
        performance, while maintaining sequential processing within each chunk.
        
        Args:
            text: Input text sequence
            
        Returns:
            List of ranks for each token (1-indexed, where 1 is most likely)
        """
        # Tokenize the input
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        if tokens.shape[1] == 0:
            return []
        
        # If batch_size is 1, use the original sequential processing
        if self.batch_size == 1:
            return self._get_token_ranks_sequential(tokens)
        
        # Use parallel batch processing
        return self._get_token_ranks_parallel(tokens)
    
    def _get_token_ranks_sequential(self, tokens: torch.Tensor) -> List[int]:
        """
        Sequential implementation of token ranking for a single batch.
        
        Args:
            tokens: Token tensor of shape [1, seq_len]
            
        Returns:
            List of ranks for each token (1-indexed, where 1 is most likely)
        """
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
    
    def _get_token_ranks_parallel(self, tokens: torch.Tensor) -> List[int]:
        """
        Parallel implementation of token ranking using model's batch processing.
        
        Args:
            tokens: Token tensor of shape [1, seq_len]
            
        Returns:
            List of ranks for each token (1-indexed, where 1 is most likely)
        """
        seq_len = tokens.shape[1]
        
        # Split tokens into chunks for parallel processing
        chunks = self._create_token_chunks(tokens, self.batch_size)
        
        if len(chunks) == 1:
            # If only one chunk, fall back to sequential processing
            return self._get_token_ranks_sequential(chunks[0])
        
        # Process all chunks in parallel
        all_ranks = []
        chunk_lengths = [chunk.shape[1] for chunk in chunks]
        max_chunk_len = max(chunk_lengths)
        
        # Pad chunks to same length for batched processing
        padded_chunks = []
        for chunk in chunks:
            if chunk.shape[1] < max_chunk_len:
                # Pad with EOS tokens
                pad_length = max_chunk_len - chunk.shape[1]
                padding = torch.full((1, pad_length), self.tokenizer.eos_token_id, 
                                   dtype=torch.long, device=self.device)
                padded_chunk = torch.cat([chunk, padding], dim=1)
            else:
                padded_chunk = chunk
            padded_chunks.append(padded_chunk)
        
        # Create batched tensor [num_chunks, max_chunk_len]
        batched_tokens = torch.cat(padded_chunks, dim=0)
        
        # Process all chunks in parallel using model's batch dimension
        # Note: Within each chunk, tokens must be processed sequentially due to
        # autoregressive nature of language models, but multiple chunks can be
        # processed simultaneously in the batch dimension
        batched_ranks = self._process_batched_tokens(batched_tokens, chunk_lengths)
        
        # Concatenate results from all chunks
        for i, chunk_ranks in enumerate(batched_ranks):
            all_ranks.extend(chunk_ranks[:chunk_lengths[i]])  # Remove padding
        
        return all_ranks
    
    def _create_token_chunks(self, tokens: torch.Tensor, num_chunks: int) -> List[torch.Tensor]:
        """
        Split tokens into chunks for parallel processing.
        
        Args:
            tokens: Token tensor of shape [1, seq_len]
            num_chunks: Number of chunks to create
            
        Returns:
            List of token tensors, one for each chunk
        """
        seq_len = tokens.shape[1]
        if seq_len == 0 or num_chunks == 1:
            return [tokens]
        
        # Calculate chunk sizes - earlier chunks get more tokens when division isn't even
        base_size = seq_len // num_chunks
        remainder = seq_len % num_chunks
        
        chunks = []
        start_idx = 0
        
        for i in range(num_chunks):
            # Earlier chunks get +1 token if there's a remainder
            chunk_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            
            if start_idx >= seq_len:
                break
                
            chunk_tokens = tokens[:, start_idx:end_idx]
            chunks.append(chunk_tokens)
            start_idx = end_idx
        
        return chunks
    
    def _process_batched_tokens(self, batched_tokens: torch.Tensor, chunk_lengths: List[int]) -> List[List[int]]:
        """
        Process multiple token sequences in parallel using model's batch processing.
        
        Args:
            batched_tokens: Tensor of shape [num_chunks, max_chunk_len]
            chunk_lengths: Actual length of each chunk (before padding)
            
        Returns:
            List of rank lists, one for each chunk
        """
        batch_size, max_seq_len = batched_tokens.shape
        all_chunk_ranks = [[] for _ in range(batch_size)]
        
        # Initialize with BOS context for all chunks
        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        bos_context = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=self.device)
        
        past_key_values = None
        
        with torch.no_grad():
            for i in range(max_seq_len):
                if i == 0:
                    # For first token, use BOS token as context for all chunks
                    outputs = self.model(bos_context, use_cache=True)
                    logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                    past_key_values = outputs.past_key_values
                else:
                    # Use previous tokens as context
                    prev_tokens = batched_tokens[:, i-1:i]  # [batch_size, 1]
                    
                    # Truncate KV cache if needed
                    past_key_values = self._truncate_kv_cache_batched(past_key_values)
                    
                    outputs = self.model(prev_tokens, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                    past_key_values = outputs.past_key_values
                
                # Process logits for each chunk
                for chunk_idx in range(batch_size):
                    # Skip if we've reached the end of this chunk
                    if i >= chunk_lengths[chunk_idx]:
                        continue
                    
                    chunk_logits = logits[chunk_idx]  # [vocab_size]
                    
                    # Get probabilities and sort by likelihood
                    probs = torch.softmax(chunk_logits, dim=-1)
                    sorted_indices = torch.argsort(probs, descending=True)
                    
                    # Find rank of actual token
                    actual_token = batched_tokens[chunk_idx, i].item()
                    rank = (sorted_indices == actual_token).nonzero(as_tuple=True)[0].item() + 1
                    all_chunk_ranks[chunk_idx].append(rank)
        
        return all_chunk_ranks
    
    def _truncate_kv_cache_batched(self, past_key_values):
        """
        Truncate KV cache for batched processing.
        
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
        Uses KV caching for efficient computation and optional batching.
        
        Args:
            ranks: List of ranks (1-indexed) for each position
            max_length: Maximum length to generate (default: len(ranks))
            
        Returns:
            Generated text string
        """
        if not ranks:
            return ""
        
        max_length = max_length or len(ranks)
        ranks = ranks[:max_length]
        
        # If batch_size is 1, use the original sequential processing
        if self.batch_size == 1:
            return self._get_string_from_token_ranks_sequential(ranks)
        
        # Use parallel batch processing
        return self._get_string_from_token_ranks_parallel(ranks)
    
    def _get_string_from_token_ranks_sequential(self, ranks: List[int]) -> str:
        """
        Sequential implementation of string generation from ranks for a single batch.
        
        Args:
            ranks: List of ranks (1-indexed) for each position
            
        Returns:
            Generated text string
        """
        tokens = self._get_tokens_from_ranks_sequential(ranks)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def _get_tokens_from_ranks_sequential(self, ranks: List[int]) -> List[int]:
        """
        Sequential implementation of token generation from ranks for a single batch.
        
        Args:
            ranks: List of ranks (1-indexed) for each position
            
        Returns:
            List of token IDs
        """
        tokens = []
        past_key_values = None
        
        with torch.no_grad():
            for i in range(len(ranks)):
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
        
        return tokens
    
    def _get_string_from_token_ranks_parallel(self, ranks: List[int]) -> str:
        """
        Parallel implementation of string generation from ranks using model's batch processing.
        
        Args:
            ranks: List of ranks (1-indexed) for each position
            
        Returns:
            Generated text string
        """
        # Split ranks into chunks for parallel processing
        rank_chunks = self._create_rank_chunks(ranks, self.batch_size)
        
        if len(rank_chunks) == 1:
            # If only one chunk, fall back to sequential processing
            return self._get_string_from_token_ranks_sequential(rank_chunks[0])
        
        # Process all chunks in parallel
        all_tokens = []
        chunk_lengths = [len(chunk) for chunk in rank_chunks]
        max_chunk_len = max(chunk_lengths)
        
        # Pad chunks to same length for batched processing
        padded_rank_chunks = []
        for chunk in rank_chunks:
            if len(chunk) < max_chunk_len:
                # Pad with rank 1 (most likely token)
                pad_length = max_chunk_len - len(chunk)
                padded_chunk = chunk + [1] * pad_length
            else:
                padded_chunk = chunk
            padded_rank_chunks.append(padded_chunk)
        
        # Process all chunks in parallel
        batched_tokens = self._process_batched_ranks(padded_rank_chunks, chunk_lengths)
        
        # Concatenate results from all chunks
        for i, chunk_tokens in enumerate(batched_tokens):
            all_tokens.extend(chunk_tokens[:chunk_lengths[i]])  # Remove padding
        
        # Decode tokens to string
        return self.tokenizer.decode(all_tokens, skip_special_tokens=True)
    
    def _create_rank_chunks(self, ranks: List[int], num_chunks: int) -> List[List[int]]:
        """
        Split ranks into chunks for parallel processing.
        
        Args:
            ranks: List of ranks
            num_chunks: Number of chunks to create
            
        Returns:
            List of rank lists, one for each chunk
        """
        seq_len = len(ranks)
        if seq_len == 0 or num_chunks == 1:
            return [ranks]
        
        # Calculate chunk sizes - earlier chunks get more ranks when division isn't even
        base_size = seq_len // num_chunks
        remainder = seq_len % num_chunks
        
        chunks = []
        start_idx = 0
        
        for i in range(num_chunks):
            # Earlier chunks get +1 rank if there's a remainder
            chunk_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            
            if start_idx >= seq_len:
                break
                
            chunk_ranks = ranks[start_idx:end_idx]
            chunks.append(chunk_ranks)
            start_idx = end_idx
        
        return chunks
    
    def _process_batched_ranks(self, batched_ranks: List[List[int]], chunk_lengths: List[int]) -> List[List[int]]:
        """
        Process multiple rank sequences in parallel using model's batch processing.
        
        Args:
            batched_ranks: List of rank lists, all padded to same length
            chunk_lengths: Actual length of each chunk (before padding)
            
        Returns:
            List of token lists, one for each chunk
        """
        batch_size = len(batched_ranks)
        max_seq_len = len(batched_ranks[0])
        all_chunk_tokens = [[] for _ in range(batch_size)]
        
        # Initialize with BOS context for all chunks
        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        bos_context = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=self.device)
        
        past_key_values = None
        
        with torch.no_grad():
            for i in range(max_seq_len):
                if i == 0:
                    # For first token, use BOS token as context for all chunks
                    outputs = self.model(bos_context, use_cache=True)
                    logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                    past_key_values = outputs.past_key_values
                else:
                    # Use previous tokens as context
                    prev_tokens = []
                    for chunk_idx in range(batch_size):
                        if i-1 < len(all_chunk_tokens[chunk_idx]):
                            prev_tokens.append(all_chunk_tokens[chunk_idx][i-1])
                        else:
                            # Use EOS if we don't have enough tokens yet
                            prev_tokens.append(self.tokenizer.eos_token_id)
                    
                    prev_tokens_tensor = torch.tensor(prev_tokens, dtype=torch.long, device=self.device).unsqueeze(1)  # [batch_size, 1]
                    
                    # Truncate KV cache if needed
                    past_key_values = self._truncate_kv_cache_batched(past_key_values)
                    
                    outputs = self.model(prev_tokens_tensor, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                    past_key_values = outputs.past_key_values
                
                # Select tokens based on ranks for each chunk
                for chunk_idx in range(batch_size):
                    # Skip if we've reached the end of this chunk
                    if i >= chunk_lengths[chunk_idx]:
                        continue
                    
                    chunk_logits = logits[chunk_idx]  # [vocab_size]
                    
                    # Get probabilities and sort by likelihood
                    probs = torch.softmax(chunk_logits, dim=-1)
                    sorted_indices = torch.argsort(probs, descending=True)
                    
                    # Select token at the specified rank (convert from 1-indexed to 0-indexed)
                    target_rank = batched_ranks[chunk_idx][i] - 1
                    if target_rank < len(sorted_indices):
                        selected_token = sorted_indices[target_rank].item()
                    else:
                        # If rank is out of bounds, use the least likely token
                        selected_token = sorted_indices[-1].item()
                    
                    all_chunk_tokens[chunk_idx].append(selected_token)
        
        return all_chunk_tokens


def rank_tokens(text: str, model_name: str = "gpt2", batch_size: int = 1) -> List[int]:
    """
    Convenience function to get token ranks for a text string.
    
    Args:
        text: Input text sequence
        model_name: HuggingFace model name
        batch_size: Number of parallel batches to process
        
    Returns:
        List of ranks for each token
    """
    ranker = LLMRanker(model_name=model_name, batch_size=batch_size)
    return ranker.get_token_ranks(text)