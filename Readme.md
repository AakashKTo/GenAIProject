---

# Project: Faster Diffusion - Rethinking the Role of the Encoder for Diffusion Model Inference

This project is for the EEE 598: Generative AI: Theory and Practice course.

## Team Members and Contributions

* **Aakash Kumar Tomar** (ASU ID: 1229632003)
    * Feature analysis, code implementation, sampling pipelines & validation.
* **Abhijeet Ghildiyal** (ASU ID: 1229612347)
    * Data setup, sampling pipelines & documentation.
* **Prateek Parashar** (ASU ID: 1229631743)
    * Result analysis, Project management, codebase maintenance & final reporting.

## Project Overview

This project implements and analyzes the "Faster Diffusion" technique, based on the paper *Rethinking the Role of the Encoder for Diffusion Model Inference*. The primary objective is to accelerate the sampling time of Stable Diffusion models without the need for retraining, while maintaining high-quality image generation.

Our main goal is to analyze the encoder and decoder features within Stable Diffusion, verifying the paper's hypothesis that encoder outputs change minimally during generation. By caching and reusing these features, we can significantly speed up the inference process.

## Methodology

### 1. Model Architecture
* **Model**: Stable Diffusion v1.5, which uses a UNet backbone and a CLIP (ViT-L/14) text encoder.
* **Samplers**: We evaluated several sampling schedulers, including DDIM, DPM-Solver, and DPM-Solver++.

### 2. Dataset
* **Dataset**: We used the validation set of the MS-COCO 2017 dataset.
* **Details**: The validation set contains 5,000 images, each paired with a single human-written caption. This dataset is a standard benchmark for text-to-image models due to its diverse and natural captions.

### 3. Faster Diffusion Implementation
 The core of this project is the implementation of **Encoder Propagation**.  The standard sampling process recomputes the full UNet at every timestep. The Faster Diffusion approach modifies this loop:
*  **Caching**: Encoder features are computed and cached only at select "Key Timesteps".
*  **Reusing**: At "Non-key Timesteps," these cached features are reused, and only the decoder half of the UNet is executed.
*  **Noise Injection**: A small amount of prior noise is injected to help recover fine texture details that might be lost during propagation.

### 4. Evaluation Metrics
To measure performance and quality, we used the following metrics:
*  **FID (Fréchet Inception Distance)**: Measures image quality and diversity.
*  **Clip Score**: Evaluates the semantic similarity between the text prompt and the generated image.
*  **Sampling Time**: The wall-clock time in seconds to generate a single image (s/image).

## Setup and Libraries

 This project focuses strictly on optimizing the inference and sampling process, not model training. The implementation relies on the following tools:

*  **Hugging Face Diffusers**: For loading the Stable Diffusion model and running sampling pipelines.
*  **PyTorch**: For all underlying tensor operations.
*  **torchmetrics**: For simplified FID computation.
*  **NumPy & Matplotlib**: For numerical analysis and plotting.
*  **scikit-learn**: For optional feature analysis.

## Results

Our implementation successfully accelerated the sampling pipeline while maintaining results very close to the baseline.

### Quantitative Results

 The following table shows a comparison of different samplers with and without our Faster Diffusion ("w/ Ours") implementation.

| Method | Steps | FID↓ | Clip score↑ | Sampling time (s)↓ |
| :--- | :--- | :--- | :--- | :--- |
| **DDIM** | 50 | 25.2524 | 0.7798 | 1.5189 |
| **DDIM w/ Ours** | 50 | 25.0416 | 0.7808 | **1.1081** |
| **DPM-Solver** | 20 | 26.9162 | 0.7781 | 2.1566 |
| **DPM-Solver w/ Ours** | 20 | 27.2168 | 0.779 | **0.6856** |
| **DPM-Solver-PP** | 20 | 27.0645 | 0.7786 | 0.6337 |
| **DPM-Solver-PP w/ Ours** | 20 | 26.8469 | 0.7806 | **0.4942** |

### Key Findings
* **What Worked**:
    *  Encoder propagation significantly reduced sampling time by approximately 24-41% compared to standard DDIM.
    *  Image quality metrics (FID and Clip score) remained very close to the baseline results.
    *  The acceleration was achieved without any model retraining, making it a highly accessible optimization.
    *  Prior noise injection was effective at recovering fine texture details.
* **Challenges**:
    *  Without noise injection, images could appear slightly smoother and lose fine textures.
    *  Aggressively skipping too many encoder steps led to "semantic drift," where images did not match prompts as closely.
    *  Manually tuning the key timesteps is required to find the right balance between speed and quality.

## Pros and Cons of the Approach

| Approach | Pros | Cons |
| :--- | :--- | :--- |
| **Faster Diffusion** |  No retraining needed, allows for parallelism, maintains good quality. |  Minor texture loss can occur without noise injection. |
| **Distillation Methods**|  Enable one-step or few-step generation. |  Require huge retraining time and result in entirely new models. |

## References
1.   Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps", 2022.
2.   Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", 2022.
3.   Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models", 2020.
4.   Song, J., Meng, C., & Ermon, S. “Denoising Diffusion Implicit Models", 2020.
5.   Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)", 2021.
6.  Lin, T.Y., Maire, M., Belongie, S., et al.  "Microsoft COCO: Common Objects in Context", 2014.
7.   Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., & Choi, Y. "CLIP Score: A Reference-free Evaluation Metric for Image Captioning", 2021.
8.   Yu, Y., Zhang, W., & Deng, Y. "Frechet Inception Distance (FID) for Evaluating GANs", 2021.
9.  Jennewein, D. M., et al. "The Sol Supercomputer at Arizona State University."  *In Practice and Experience in Advanced Research Computing*, 2023.
