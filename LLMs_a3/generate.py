import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # Creating a copy of the input tensor to append the generated tokens
        all_ids = input_ids.clone()

        # Initializing a list to store the new generated tokens
        tokens_generated = []  

        # Disable gradient computation for inference
        with torch.no_grad():  
            for z in range(self.max_output_len):
                
                # Giving all the ids to the model to generate the output
                outputs = self.model(all_ids)  #  outputs shape: (1, seq_len, vocab_size)   # Reference: https://discuss.huggingface.co/t/questions-about-the-shape-of-t5-logits/10207

                # Get the logits for the last token
                logits = outputs.logits[:, -1, :]   # logits shape: (1, vocab_size) 

                # Select the token with the highest probability
                next_token = torch.argmax(logits, dim=-1)  # next_token shape: tensor((1,))

                # Convert it to scalar
                next_token = next_token.item()  

                # Append the generated token to the list tokens_generated
                tokens_generated.append(next_token)

                # Break the loop if the <EOS> token is generated
                if next_token == self.eos_token_id:
                    break

                # Converting the new token generated such that we add it to the input token list
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=all_ids.device)

                # Update the input by appending the new token
                all_ids = torch.cat([all_ids, next_token_tensor], dim=1)  # Shape: (1, P+generated)

        # Convert the list of generated tokens to a 1D tensor
        return torch.tensor(tokens_generated, dtype=torch.long, device=all_ids.device)
        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # Creating a copy of the input tensor to append the generated tokens
        all_ids = input_ids.clone()

        # Initializing a list to store the generated tokens (more efficient than tensor concatenation)
        tokens_generated = []

        # Disable gradient computation for inference
        with torch.no_grad():
            for _ in range(self.max_output_len):
                
                # Giving all the ids to the model to generate the output
                outputs = self.model(all_ids)  # outputs shape: (1, seq_len, vocab_size)
                # Reference: https://discuss.huggingface.co/t/questions-about-the-shape-of-t5-logits/10207

                # Get the logits for the last token
                logits = outputs.logits[:, -1, :]  # logits shape: (1, vocab_size) 

                # Applying temperature scaling to logits values
                scaled_logits = logits / self.tau

                # Converting the logits to probabilities using the softmax function
                probs = torch.softmax(scaled_logits, dim=-1)  # Shape: (1, vocab_size)

                # Sampling the next token from the probability distribution
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1).item()  # Convert to scalar

                # Append the generated token to the list `tokens_generated`
                tokens_generated.append(next_token)

                # Break the loop if the <EOS> token is generated
                if next_token == self.eos_token_id:
                    break

                # Update the input by appending the new token
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=all_ids.device)
                all_ids = torch.cat([all_ids, next_token_tensor], dim=1)  # Shape: (1, P+generated)

        # Convert the list of generated tokens to a tensor once at the end (efficient)
        return torch.tensor(tokens_generated, dtype=torch.long, device=all_ids.device)
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # Creating a copy of the input tensor to append the generated tokens
        all_ids = input_ids.clone()  # Shape: (1, P)

        # Initializing a list to store generated tokens
        tokens_generated = []

        # Disable gradient computation for inference
        with torch.no_grad():
            for _ in range(self.max_output_len):
                # Giving all the ids to the model to generate the output
                outputs = self.model(all_ids)  # Shape: (1, seq_len, vocab_size)

                # Get the logits for the last token
                logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)

                # Get the top-k logits and their indices
                topk_logits, topk_indices = torch.topk(logits, k=self.k, dim=-1)  # Shapes: (1, k), (1, k)

                # Convert top-k logits to probabilities using softmax
                topk_probs = torch.softmax(topk_logits, dim=-1)  # Shape: (1, k)

                # Sample from the top-k probabilities
                sampled_index = torch.multinomial(topk_probs, num_samples=1).item()  # Convert tensor to scalar

                # Map the sampled index back to the original token ID
                next_token = topk_indices[0, sampled_index].item()

                # Append the generated token to the list `tokens_generated`
                tokens_generated.append(next_token)

                # Break the loop if the <EOS> token is generated
                if next_token == self.eos_token_id:
                    break

                # Update the input by appending the new token
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=all_ids.device)
                all_ids = torch.cat([all_ids, next_token_tensor], dim=1)

        # Convert the list of generated tokens to a tensor once at the end (efficient)
        return torch.tensor(tokens_generated, dtype=torch.long, device=all_ids.device)
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # Initializing a list to store the new generated tokens
        generated_tokens = []

        # Creating a copy of the input tensor to append the generated tokens
        current_input_ids = input_ids.clone()

        # Disable gradient computation for inference
        with torch.no_grad():
            for _ in range(self.max_output_len):
                # Giving all the ids to the model to generate the output
                outputs = self.model(current_input_ids)  # outputs shape: (1, seq_len, vocab_size)

                # Get the logits for the last token
                logits = outputs.logits[0, -1, :]  # logits shape: (vocab_size)

                # Sorting logits in descending order and retrieving corresponding token indices
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # Shapes: (vocab_size), (vocab_size)

                # Converting the logits to probabilities using the softmax function with temperature
                probabilities = torch.nn.functional.softmax(sorted_logits / self.tau, dim=-1)  # Shape: (vocab_size)

                # Compute cumulative probabilities
                cumulative_probabilities = torch.cumsum(probabilities, dim=-1)  # Shape: (vocab_size)

                # Create mask for top-p probabilities
                mask = cumulative_probabilities < self.p  # Shape: (vocab_size)

                # If no tokens meet the threshold, take at least one token
                if mask.sum() == 0:
                    top_p_index_to_keep = 0
                else:
                    # Finding the last index where cumulative probability is below self.p
                    top_p_index_to_keep = torch.where(mask)[0][-1].item()

                # Selecting indices to remove beyond top-p and setting their logits to -inf
                indices_to_remove = sorted_indices[top_p_index_to_keep + 1:]
                sorted_logits[indices_to_remove] = float('-inf')

                # Converting filtered logits back to probabilities for sampling
                probabilities = torch.nn.functional.softmax(sorted_logits / self.tau, dim=-1)  # Shape: (vocab_size)

                # Sampling the next token from the filtered distribution
                next_token = torch.multinomial(probabilities, num_samples=1)  # Shape: (1)
                next_token_id = sorted_indices[next_token].item()  # Get the actual token ID

                # Append the generated token to the list generated_tokens
                generated_tokens.append(next_token_id)

                # Break the loop if the <EOS> token is generated
                if next_token_id == self.eos_token_id:
                    break

                # Converting the new token generated such that we add it to the input token tensor
                next_token_tensor = torch.tensor([[next_token_id]], device=current_input_ids.device)

                # Update the input by appending the new token
                current_input_ids = torch.cat(
                    [current_input_ids, next_token_tensor], 
                    dim=1
                )  # Shape: (1, P+generated)

        # Convert the list of generated tokens to a 1D tensor
        return torch.tensor(generated_tokens, device=input_ids.device)