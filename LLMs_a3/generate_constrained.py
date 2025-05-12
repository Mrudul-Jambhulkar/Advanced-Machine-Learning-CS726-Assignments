#references : (1) chatgpt   (2)https://www.geeksforgeeks.org/trie-insert-and-search/  (3)https://pytorch.org/docs/stable/index.html
import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

#reference : (1) chatgpt   (2)https://www.geeksforgeeks.org/trie-insert-and-search/
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        """Insert a word into the trie with proper space handling"""
        # Normalize whitespace and handle subwords
        word = " " + word.strip()  # Ensure leading space for word boundaries
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search_prefix(self, prefix: str):
        """Search for a prefix in the trie with space handling"""
        # Ensure consistent space handling
        if not prefix.startswith(" "):
            prefix = " " + prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node


class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tokenizer = tokenizer
    
    def _build_word_trie(self, word_list: list) -> Trie:
        """Build a Trie from the word list, preserving original words with space handling"""
        trie = Trie()
        for word in word_list:
            trie.insert(word.strip())
        return trie

#reference : (1) chatgpt (2)https://pytorch.org/docs/stable/index.html
    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
        word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Generate constrained text using the provided word list.
            
            input_ids: Input token IDs
            word_list: List of allowed words
            
            Returns: Generated token IDs
        '''
        # Initialize
        all_ids = input_ids.clone()  # Shape: (1, in_seq_len)
        tokens_generated = []
        word_trie = self._build_word_trie(word_list)
        current_prefix = " "  # Initialize with space for word boundaries

        with torch.no_grad():
            for _ in range(self.max_output_len):
                # Get logits for the next token
                outputs = self.model(all_ids)
                logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)

                # Mask invalid tokens
                valid_mask = torch.zeros_like(logits, dtype=torch.bool)
                for token_id in range(logits.shape[-1]):
                    # Decode the potential next token (preserving spaces/special chars)
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    new_prefix = current_prefix + token_text
                    prefix_node = word_trie.search_prefix(new_prefix)
                    # Valid if prefix exists and leads to a word
                    if prefix_node is not None:
                        valid_mask[0, token_id] = True

                # Apply mask to logits
                masked_logits = torch.full_like(logits, -float('inf'))   #https://pytorch.org/docs/stable/generated/torch.full_like.html
                constrained_logits = torch.where(valid_mask, logits, masked_logits) #https://pytorch.org/docs/stable/generated/torch.where.html
                next_token = torch.argmax(constrained_logits, dim=-1).item()

                # Check for invalid generation or EOS
                if constrained_logits[0, next_token] == -float('inf'):
                    break
                if next_token == self.eos_token_id:
                    tokens_generated.append(next_token)
                    break

                # Update state
                tokens_generated.append(next_token)
                token_text = self.tokenizer.decode([next_token], skip_special_tokens=False)
                current_prefix += token_text

                # Reset prefix if a complete word is formed (maintain leading space)
                prefix_node = word_trie.search_prefix(current_prefix)
                if prefix_node and prefix_node.is_end:
                    current_prefix = " "  # Reset with space for next word

                # Append to input for next iteration
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=all_ids.device)
                all_ids = torch.cat([all_ids, next_token_tensor], dim=1)

        return torch.tensor(tokens_generated, dtype=torch.long, device=all_ids.device)
        
 #BLEU:0.28797250859106527
# ROUGE-1: 0.3218666911645693
# ROUGE-2: 0.16630608773713146
# ROUGE-LCS: 0.2816582818401717