# CS726 Assignment 2 - Denoising Diffusion Probabilistic Models (DDPM)

This repository contains the implementation of **unconditional and conditional DDPMs** (Denoising Diffusion Probabilistic Models) as part of CS726 Assignment 2. The goal is to train generative models capable of synthesizing high-quality images using a diffusion-based approach. This implementation includes key features such as configurable noise schedules, classifier-free guidance, and both forward and reverse diffusion processes.

## Overview 

Diffusion models are a class of generative models that learn to reverse a gradual noising process to generate data from pure noise. The model is trained to predict the noise added at each step of a forward diffusion process, and during sampling, it denoises step by step starting from random Gaussian noise.

In this project:
- We implement **unconditional DDPM** and **Conditional DDPM** (using class labels).
- Noise schedules can be linear or cosine-based.
- Classifier-free guidance is explored to improve conditional generation without relying on a separate classifier.


## Files

- `ddpm.py`  
  Implements:
  - `NoiseScheduler` with forward and reverse diffusion methods
  - `DDPM` model (a U-Net-style convolutional architecture for MNIST)
  - `ConditionalDDPM` (adds class-conditioning through class embeddings)
  - Training and sampling procedures
  - Classifier-Free Guidance for conditional sampling

- `ddpm_noise.py`  
  Contains alternative noise schedule strategies (e.g., cosine noise, linear noise) used for training and sampling.

- `CS726_Assignment_2_report.pdf`  
  A comprehensive report describing:
  - Mathematical background of DDPMs
  - Implementation details
  - Experiment results (e.g., effect of number of diffusion steps, guidance scale)
  - Generated image samples
  - Observations and analysis

## Highlights

- ✅ Forward and reverse diffusion defined with closed-form variances and means  
- ✅ Learned noise prediction using simple CNN (inspired by U-Net)  
- ✅ Configurable training for conditional or unconditional settings  
- ✅ Support for different noise schedules  
- ✅ Classifier-free guidance for improved class-conditional generation

## Setup

Install the required packages from requirements.txt
