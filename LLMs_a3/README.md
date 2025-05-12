# CS726 Programming Assignment 3: LLM Decoding Techniques

This repository contains the implementation for **Programming Assignment 3** of the CS726: Advanced Machine Learning course (Spring 2025). The assignment focuses on exploring decoding techniques for Large Language Models (LLMs), specifically Llama-2, on the Hindi-to-English translation task using the IN22-Gen dataset. This README covers **Task 0** (Introduction to LLM Decoding Techniques) and **Task 1** (Word-Constrained Decoding).



## Table of Contents

- [Overview](#overview)
- [Tasks](#tasks)



## Overview

This assignment implements and analyzes various decoding strategies for LLMs, focusing on their impact on text generation for Hindi-to-English translation. The tasks are implemented in Python using PyTorch and evaluated using BLEU and ROUGE scores. The code is structured to run within a provided environment, and the scripts `task0.py` and `task1.py` execute the core decoding logic in `generate.py` and `generate_constrained.py`, respectively.

## Tasks

1. **Task 0: Introduction to LLM Decoding Techniques** 
   - Implement and compare four decoding strategies for Llama-2 on the IN22-Gen dataset:
     - **Greedy Decoding**: Selects the most probable token at each step.
     - **Random Sampling with Temperature Scaling**: Samples tokens with temperature adjustments (τ = YAN0.5, 0.9).
     - **Top-k Sampling**: Samples from the top k most probable tokens (k = 5, 10).
     - **Nucleus Sampling**: Samples from the smallest set of tokens with cumulative probability ≥ p (p = 0.5, 0.9).
   - Evaluate generated translations using BLEU and ROUGE scores.
   - Script: `task0.py` runs `generate.py` to perform decoding and evaluation.

2. **Task 1: Word-Constrained Decoding** 
   - Implement a greedy decoding variant that uses a provided bag of words (word_lists.txt) to constrain the output, leveraging tokenization properties of the LLM.
   - Use a Trie-based approach to ensure generated tokens align with the word list.
   - Compare performance against Task 0 strategies using BLEU and ROUGE scores.
   - Script: `task1.py` runs `generate_constrained.py` to perform constrained decoding and evaluation.

