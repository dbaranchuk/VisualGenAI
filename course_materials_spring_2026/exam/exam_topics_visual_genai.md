---
# Visual GenAI | Exam Topics

---

## 1. Divergences
- KL divergence and Fisher divergence  
- Forward KL vs Reverse KL: mode-covering vs mode-seeking behavior  

## 2. Diffusion Model Formulations

### 2.1 Discrete-Time Variational Diffusion (DDPM)
- Denoising Diffusion Probabilistic Models formulation  
- Training objective and ELBO derivation  
- Conditional trick  
- Sampling procedure  
- Variance-preserving (VP) schedules  
- Connection to VAEs  

### 2.2 Discrete-Time Score-Based Diffusion (DSM)
- Denoising Score Matching formulation  
- Training objective and optimal solution  
- Conditional trick  
- Tweedie’s formula. Ideal Denoiser  
- Sampling procedure  
- Motivation for multiple noise levels  
- Variance-exploding (VE) schedules  

### 2.3 Continuous-Time Diffusion (SDE/ODE)
- SDE and ODE formulations  
- Wiener process (Brownian motion)  
- Fokker–Planck and continuity equations. Why do we need them?  
- Continuous-time schedule derivation  

### 2.4 Flow Matching (FM)
- FM formulation
- Training objective and optimal solution  
- Conditional trick  
- Sampling procedure  
- Linear schedule  
- Advantages over continuous diffusion formulations  

## 3. Sampling Methods

### 3.1 Numerical Solvers
- Euler, Heun  
- DDIM, [DPM-Solver](https://arxiv.org/abs/2206.00927)  
- Single-step vs multi-step (Adams–Bashforth)  
- Connections across diffusion formulations  

### 3.2 Timestep Schedules
- Linear and cosine schedules  
- EDM and SD3 schedules  
- Shift selection strategies=

### 3.3 Guidance Methods
- Classifier guidance
- [Classifier-free guidance](https://arxiv.org/abs/2207.12598)  
- [Interval guidance](https://arxiv.org/abs/2404.07724)  
- [AutoGuidance](https://arxiv.org/pdf/2406.02507v1)  


## 4. Training Design Choices

### 4.1 Parameterizations and Losses
- ε-, x₀-, and v-prediction  
- Conversion between parameterizations  
- Corresponding loss functions  

(Table 1. [JiT](https://arxiv.org/pdf/2511.13720) but for the 1 --> 0 process)

### 4.2 Timestep Sampling
- Training timestep distributions (Uniform, Logit-normal)
- Resolution-dependent shift selection  

### 4.3 Representation Alignment
- [REPA](https://arxiv.org/abs/2410.06940) and [iREPA](https://arxiv.org/abs/2512.10794)  


## 5. Architectures and Conditioning

### 5.1 Core Architectures
- UNet ([SDXL](https://arxiv.org/abs/2307.01952))
- Diffusion Transformers ([DiT](https://arxiv.org/abs/2212.09748))  
- Multimodal DiT ([MM-DiT](https://arxiv.org/abs/2403.03206))  

### 5.2 Conditioning Mechanisms
- Timestep conditioning  
- Class and text conditioning  
- Adapter-based conditioning:
  - [ControlNet](https://arxiv.org/abs/2302.05543)  
  - [IP-Adapter](https://arxiv.org/abs/2308.06721)  

### 5.3 Latent vs Pixel-Space Diffusion
- Latent diffusion models  
- Trade-offs: learnability vs reconstruction quality  
- Representation Autoencoders ([RAE](https://arxiv.org/abs/2510.11690))  


## 6. Few-step Models

*Expectations:* 
- Training and sampling procedures
- Pros and cons for each variant
- Training from scratch vs distillation 
- High-level connections between flow-map approaches
- Key difference between distribution matching and flow-map approaches

### 6.1 Flow-Map Models
- General formulation  
- Knowledge distillation  
- Consistency models:
  - Discrete-time and continuous-time [Consistency Models](https://arxiv.org/abs/2303.01469)
  - [Multi-boundary(-step) Consistency](https://arxiv.org/abs/2403.06807)
  - Consistency Trajectory Models ([CTM](https://openreview.net/forum?id=ymjI8feDTD))  
- [Shortcut models](https://arxiv.org/abs/2410.12557) and [Mean Flow](https://arxiv.org/abs/2505.13447)

### 6.2 Distribution Matching

- Methods: [DMD](https://arxiv.org/abs/2311.18828)/[DMD2](https://arxiv.org/abs/2405.14867), [ADD](https://arxiv.org/abs/2311.17042)/[LADD](https://arxiv.org/abs/2403.12015), [MMD](https://arxiv.org/pdf/2503.16397)

## 7. Autoregressive Models

*Expectations:* understanding training and sampling procedures, and pros and cons for each variant.

- Image tokenizers:
  - [VQ-VAE](https://arxiv.org/pdf/1711.00937) (w/o PixelCNN)  
  - [VQ-GAN](https://arxiv.org/pdf/2012.09841)  
  - [ViT-VQ-GAN](https://arxiv.org/abs/2110.04627)  
- Prediction paradigms:
  - Next-token prediction ([LlamaGen](https://arxiv.org/abs/2406.06525))
  - Next-scale prediction ([VAR](https://arxiv.org/abs/2404.02905))  
  - Continuous-token AR ([MAR](https://arxiv.org/abs/2406.11838))  
- [Diffusion as implicit spectral autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html)  


## 8. Extensions Beyond Images

*Expectations:* understanding high-level model designs and ideas, and their pros and cons.

### 8.1 Video Diffusion
- Architectural differences from image models  
- Frame-autoregressive diffusion
  - Teacher forcing vs [Diffusion forcing](https://openreview.net/forum?id=yDo1ynArjj) vs [Self forcing](https://arxiv.org/abs/2506.08009)  

### 8.2 Multimodal Models
- Multi-Modal Large Language Models (MLLM), aka pure AR models for text and images
    - Discrete image tokenizer ([Janus-Pro](https://arxiv.org/pdf/2501.17811))
    - Continuous image tokenizer ([UniFluid](https://arxiv.org/abs/2503.13436))
- Unified AR for text + diffusion for images ([Bagel](https://arxiv.org/abs/2505.14683), [TransFusion](https://arxiv.org/abs/2408.11039))
- VLM encoder + diffusion decoder ([Qwen-image](https://arxiv.org/abs/2508.02324)) 

### 8.3 3D Generative Models
- 2D diffusion for 3D training (DMD-like for 3D):
  - Score Distillation Sampling ([SDS](https://arxiv.org/pdf/2209.14988)) 
  - Variational Score Distillation ([VSD](https://arxiv.org/abs/2305.16213))
- Multi-view diffusion architectures ([SEVA](https://arxiv.org/pdf/2503.14489))  


---

# Materials

- [Course Materials](https://github.com/dbaranchuk/VisualGenAI)
- [Tracing the Principles Behind Modern Diffusion Models](https://the-principles-of-diffusion-models.github.io/#/blog)
- [FLUX.2 blogpost](https://bfl.ai/techblog/representation-comparison) about VAE latent spaces and timestep shifts
