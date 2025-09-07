Of course. Here is a README file created from the provided presentation.

---

# Project: Faster Diffusion - Rethinking the Role of the Encoder for Diffusion Model Inference

[cite_start]This project is for the EEE 598: Generative AI: Theory and Practice course[cite: 1].

## Team Members and Contributions

* **Aakash Kumar Tomar** (ASU ID: 1229632003)
    * [cite_start]Feature analysis, code implementation, sampling pipelines & validation[cite: 125].
* **Abhijeet Ghildiyal** (ASU ID: 1229612347)
    * [cite_start]Data setup, sampling pipelines & documentation[cite: 128].
* **Prateek Parashar** (ASU ID: 1229631743)
    * [cite_start]Result analysis, Project management, codebase maintenance & final reporting[cite: 131].

## Project Overview

[cite_start]This project implements and analyzes the "Faster Diffusion" technique, based on the paper *Rethinking the Role of the Encoder for Diffusion Model Inference*[cite: 5, 7]. [cite_start]The primary objective is to accelerate the sampling time of Stable Diffusion models without the need for retraining, while maintaining high-quality image generation[cite: 32, 33].

[cite_start]Our main goal is to analyze the encoder and decoder features within Stable Diffusion, verifying the paper's hypothesis that encoder outputs change minimally during generation[cite: 28]. [cite_start]By caching and reusing these features, we can significantly speed up the inference process[cite: 50].

## Methodology

### 1. Model Architecture
* [cite_start]**Model**: Stable Diffusion v1.5, which uses a UNet backbone and a CLIP (ViT-L/14) text encoder[cite: 11, 76].
* [cite_start]**Samplers**: We evaluated several sampling schedulers, including DDIM, DPM-Solver, and DPM-Solver++[cite: 12, 77].

### 2. Dataset
* [cite_start]**Dataset**: We used the validation set of the MS-COCO 2017 dataset[cite: 14, 34].
* [cite_start]**Details**: The validation set contains 5,000 images, each paired with a single human-written caption[cite: 39, 42]. [cite_start]This dataset is a standard benchmark for text-to-image models due to its diverse and natural captions[cite: 36].

### 3. Faster Diffusion Implementation
[cite_start]The core of this project is the implementation of **Encoder Propagation**[cite: 61]. [cite_start]The standard sampling process recomputes the full UNet at every timestep[cite: 59]. The Faster Diffusion approach modifies this loop:
* [cite_start]**Caching**: Encoder features are computed and cached only at select "Key Timesteps"[cite: 50, 54].
* [cite_start]**Reusing**: At "Non-key Timesteps," these cached features are reused, and only the decoder half of the UNet is executed[cite: 50, 55].
* [cite_start]**Noise Injection**: A small amount of prior noise is injected to help recover fine texture details that might be lost during propagation[cite: 101, 106, 120].

### 4. Evaluation Metrics
To measure performance and quality, we used the following metrics:
* [cite_start]**FID (Fréchet Inception Distance)**: Measures image quality and diversity[cite: 16].
* [cite_start]**Clip Score**: Evaluates the semantic similarity between the text prompt and the generated image[cite: 19].
* [cite_start]**Sampling Time**: The wall-clock time in seconds to generate a single image (s/image)[cite: 20].

## Setup and Libraries

[cite_start]This project focuses strictly on optimizing the inference and sampling process, not model training[cite: 88]. The implementation relies on the following tools:

* [cite_start]**Hugging Face Diffusers**: For loading the Stable Diffusion model and running sampling pipelines[cite: 80].
* [cite_start]**PyTorch**: For all underlying tensor operations[cite: 81].
* [cite_start]**torchmetrics**: For simplified FID computation[cite: 83].
* [cite_start]**NumPy & Matplotlib**: For numerical analysis and plotting[cite: 82].
* [cite_start]**scikit-learn**: For optional feature analysis[cite: 81].

## Results

Our implementation successfully accelerated the sampling pipeline while maintaining results very close to the baseline.

### Quantitative Results

[cite_start]The following table shows a comparison of different samplers with and without our Faster Diffusion ("w/ Ours") implementation[cite: 114].

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
    * [cite_start]Encoder propagation significantly reduced sampling time by approximately 24-41% compared to standard DDIM[cite: 100].
    * [cite_start]Image quality metrics (FID and Clip score) remained very close to the baseline results[cite: 102, 118].
    * [cite_start]The acceleration was achieved without any model retraining, making it a highly accessible optimization[cite: 104, 117].
    * [cite_start]Prior noise injection was effective at recovering fine texture details[cite: 101].
* **Challenges**:
    * [cite_start]Without noise injection, images could appear slightly smoother and lose fine textures[cite: 106].
    * [cite_start]Aggressively skipping too many encoder steps led to "semantic drift," where images did not match prompts as closely[cite: 107].
    * [cite_start]Manually tuning the key timesteps is required to find the right balance between speed and quality[cite: 121].

## Pros and Cons of the Approach

| Approach | Pros | Cons |
| :--- | :--- | :--- |
| **Faster Diffusion** | [cite_start]No retraining needed, allows for parallelism, maintains good quality[cite: 109]. | [cite_start]Minor texture loss can occur without noise injection[cite: 109, 120]. |
| **Distillation Methods**| [cite_start]Enable one-step or few-step generation[cite: 109]. | [cite_start]Require huge retraining time and result in entirely new models[cite: 109]. |

## References
1.  [cite_start]Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps", 2022[cite: 142].
2.  [cite_start]Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", 2022[cite: 139].
3.  [cite_start]Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models", 2020[cite: 141].
4.  [cite_start]Song, J., Meng, C., & Ermon, S. “Denoising Diffusion Implicit Models", 2020[cite: 140].
5.  [cite_start]Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)", 2021[cite: 136].
6.  Lin, T.Y., Maire, M., Belongie, S., et al. [cite_start]"Microsoft COCO: Common Objects in Context", 2014[cite: 135].
7.  [cite_start]Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., & Choi, Y. "CLIP Score: A Reference-free Evaluation Metric for Image Captioning", 2021[cite: 137].
8.  [cite_start]Yu, Y., Zhang, W., & Deng, Y. "Frechet Inception Distance (FID) for Evaluating GANs", 2021[cite: 138].
9.  Jennewein, D. M., et al. "The Sol Supercomputer at Arizona State University." [cite_start]*In Practice and Experience in Advanced Research Computing*, 2023[cite: 133].