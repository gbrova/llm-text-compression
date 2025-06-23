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
    
    def get_token_ranks(self, text: str) -> List[int]:
        """
        Get the rank of each token in the sequence given previous context.
        
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
        
        with torch.no_grad():
            for i in range(tokens.shape[1]):  # Start from 0 to include first token
                if i == 0:
                    # For first token, use empty context (beginning of sequence)
                    context = torch.empty((1, 0), dtype=torch.long).to(self.device)
                else:
                    # Get context up to current position (respecting max context length)
                    start_idx = max(0, i - self.max_context_length)
                    context = tokens[:, start_idx:i]
                
                # Get model predictions for next token
                outputs = self.model(context)
                logits = outputs.logits[0, -1, :] if context.shape[1] > 0 else outputs.logits[0, 0, :]
                
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