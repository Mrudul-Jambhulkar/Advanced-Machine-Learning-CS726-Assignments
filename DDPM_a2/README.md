# CS726 Assignment 2 - Denoising Diffusion Probabilistic Models (DDPM)

This repository contains the implementation of **unconditional and conditional DDPMs** (Denoising Diffusion Probabilistic Models) as part of CS726 Assignment 2. The goal is to train generative models capable of synthesizing high-quality images using a diffusion-based approach. This implementation includes key features such as configurable noise schedules, classifier-free guidance, and both forward and reverse diffusion processes.

## Overview
This project provides a modular implementation of DDPM and Conditional DDPM for generating synthetic data. Key components include:

- **NoiseScheduler**: Manages the noise schedule (linear beta schedule) for forward and reverse diffusion processes.
- **DDPM**: A neural network for predicting noise in the unconditional diffusion process.
- **ConditionalDDPM**: Extends DDPM to condition the generation on class labels.
- **ClassifierDDPM**: Uses the ConditionalDDPM model for classification based on likelihood estimation.
- **Training and Sampling**: Functions to train the models and generate samples, with support for classifier-free guidance (CFG) in ConditionalDDPM.
- **Evaluation and Visualization**: Metrics (NLL, EMD) and visualizations to compare real and generated samples.

The code supports 2D and 3D datasets and includes utilities for loading datasets, evaluating samples, and saving results.


## Files

- `ddpm.py`  
  Implements:
  - `NoiseScheduler` with forward and reverse diffusion methods
  - `DDPM` model 
  - `ConditionalDDPM` (adds class-conditioning through class embeddings)
  - Training and sampling procedures
  - Classifier-Free Guidance for conditional sampling

- `ddpm_noise.py`  
  Contains alternative noise schedule strategies (e.g., cosine noise, linear noise) used for training and sampling.

- `CS726_Assignment_2_report.pdf`  
  A comprehensive report describing:
  - Implementation details
  - Experiment results (e.g., effect of number of diffusion steps, guidance scale)
  - Generated image samples
  - Observations and analysis


## Setup

Install the required packages from requirements.txt
