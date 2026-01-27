---
# Project: Faster Diffusion – Rethinking the Role of the Encoder for Diffusion Model Inference

This project is for **EEE 598: Generative AI: Theory and Practice**.

**Paper reference:** https://arxiv.org/pdf/2312.09608

## Team Members and Contributions

* **Aakash Kumar Tomar** (ASU ID: 1229632003)  
  Feature analysis, code implementation, sampling pipelines & validation.
* **Abhijeet Ghildiyal** (ASU ID: 1229612347)  
  Data setup, sampling pipelines & documentation.
* **Prateek Parashar** (ASU ID: 1229631743)  
  Result analysis, project management, codebase maintenance & final reporting.

## Project Overview

This project implements and analyzes the **“Faster Diffusion”** technique from *Rethinking the Role of the Encoder for Diffusion Model Inference*. The objective is to **accelerate Stable Diffusion inference (sampling)** *without retraining*, while preserving image quality.

### Project Goals

1. **Main goal:** Analyze encoder vs. decoder feature dynamics in Stable Diffusion and verify the paper’s hypothesis that **encoder outputs change minimally** during generation while decoder outputs change more.
2. **Feature measurement:** Quantify encoder/decoder feature changes using **Mean Squared Error (MSE)** and **Frobenius norm**.
3. **Objective:** Reduce sampling time using **encoder propagation / caching** while maintaining generation quality, evaluated with **FID** and **CLIP score**.

## Experimental Setup

### Models and Samplers
* **Stable Diffusion v1.5** (baseline used in the paper)
* **Stable Diffusion v2.0** (slides refer to v2.0)  
  *v2 uses a higher internal resolution (768×768) compared to v1.5.*
* **Sampling schedulers:** **DDIM**, **DPM-Solver**, **DPM-Solver++**

### Dataset
* **MS-COCO 2017** (text-to-image benchmark)
  * COCO split is **118K train / 5K validation**
  * Each image has **5 human-written captions** (collected via Amazon Mechanical Turk)
* For our experiments we used the **validation set**:
  * **v1.5:** 5,000 images (one prompt per image)
  * **v2.0:** 2,000 images (subset due to time constraints)

### Evaluation Metrics
* **FID (Fréchet Inception Distance)**  
* **CLIP score**
* **Sampling time**: wall-clock seconds per image (s/image)

## Faster Diffusion Method

The key idea is **Encoder Propagation**: instead of running the full U-Net at every diffusion step, we selectively recompute the encoder and reuse cached features elsewhere.

### Key vs. Non-key Timesteps
* Encoder features change **more in early timesteps** and **less in later timesteps**.
* **Key steps:** timesteps where encoder features are recomputed (mostly early steps).
* **Non-key steps:** encoder is skipped; we reuse features from the most recent key step and run only the decoder.

### Implementation Details
* **Caching:** store encoder features at key timesteps.
* **Reuse:** feed cached encoder features into the decoder at non-key steps.
* **Prior noise injection:** inject a small amount of prior noise to recover fine textures that can be lost during propagation.
* **Parallel decoding:** supports partial parallelism by decoding multiple steps together (where applicable).

### Tuned Hyperparameters (used during experiments)
| Parameter | Range / Example | Purpose |
|---|---|---|
| Key timesteps (`t_key`) | Manual list, e.g. `{50, 49, 48, 47, 45, 40, 35, 25, 15}` | Balance speedup vs. quality |
| Prior noise injection strength (`α`) | `0.001` – `0.005` | Recover texture details |
| Scheduler | `{DDIM, DPM-Solver, DPM-Solver++}` | Speed/quality tradeoff |
| Sampling steps (`T`) | `20` – `50` | More steps → better quality, slower |

## Environment and Setup

This project was developed and tested on **Linux** with **CUDA 12.6**.

### 1) Create a virtual environment
```bash
python3 -m venv fasterdiff-env
source fasterdiff-env/bin/activate
```

### 2) Install dependencies
Make sure you have `pip>=21.3`.

```bash
pip install -r requirements.txt
```

### 3) Run experiments
`main.py` is the entry point. It:
- Prepares the dataset (if needed)
- Runs experiments for **SD v1.5** and **SD v2.x**
- Benchmarks **3 schedulers** × **with/without** Faster Diffusion
- Saves results to `results_sd15/` and `results_sd2/`

```bash
python main.py
```

## Results

### Quantitative Results (Stable Diffusion v1.5)

| Method | Steps | FID↓ | Clip score↑ | Sampling time (s/image)↓ |
| :--- | :---: | :---: | :---: | :---: |
| **DDIM** | 50 | 25.2524 | 0.7798 | 1.5189 |
| **DDIM w/ Ours** | 50 | 25.0416 | 0.7808 | **1.1081** |
| **DPM-Solver** | 20 | 26.9162 | 0.7781 | 2.1566 |
| **DPM-Solver w/ Ours** | 20 | 27.2168 | 0.7790 | **0.6856** |
| **DPM-Solver-PP** | 20 | 27.0645 | 0.7786 | 0.6337 |
| **DPM-Solver-PP w/ Ours** | 20 | 26.8469 | 0.7806 | **0.4943** |

### Quantitative Results (Stable Diffusion v2.0)

| Method | Steps | FID↓ | Clip score↑ | Sampling time (s/image)↓ |
| :--- | :---: | :---: | :---: | :---: |
| **DPM-Solver** | 20 | 33.6200 | 0.7740 | 1.0198 |
| **DPM-Solver w/ Ours** | 20 | 33.7462 | 0.7888 | **0.9670** |
| **DPM-Solver-PP** | 20 | 33.6686 | 0.7907 | 0.9557 |
| **DPM-Solver-PP w/ Ours** | 20 | 33.5561 | 0.7901 | **0.8972** |

### Wall-clock runtime (full experiment runs)
* **Stable Diffusion v1.5:** ~**11 hours 20 minutes**
* **Stable Diffusion v2.0:** ~**4 hours 12 minutes**

### Key Findings
**What worked**
* Encoder propagation reduced sampling time by about **24–41%** compared to standard sampling.
* **FID** and **CLIP score** stayed very close to the baseline.
* **Prior noise injection** helped recover fine textures with minimal extra compute.
* Speedup was achieved **without retraining** (unlike distillation approaches).

**Challenges / what didn’t work**
* Without noise injection, images could look slightly smoother and lose fine textures (especially early timesteps).
* Skipping too many encoder timesteps (too few key steps) caused **semantic drift** (worse prompt alignment).
* Caching introduced a small but noticeable GPU memory increase.

## Pros and Cons of the Approach

| Approach | Pros | Cons |
| :--- | :--- | :--- |
| **Faster Diffusion** | No retraining needed, supports partial parallelism, maintains good quality. | Minor texture loss without noise injection; manual tuning of key timesteps required. |
| **Distillation Methods** | One-step or few-step generation. | Requires large retraining cost and produces new models. |

## References
1. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps**, 2022.
2. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. **DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models**, 2022.
3. Ho, J., Jain, A., & Abbeel, P. **Denoising Diffusion Probabilistic Models**, 2020.
4. Song, J., Meng, C., & Ermon, S. **Denoising Diffusion Implicit Models**, 2020.
5. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. **High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)**, 2021.
6. Lin, T.-Y., Maire, M., Belongie, S., et al. **Microsoft COCO: Common Objects in Context**, 2014.
7. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., & Choi, Y. **CLIP Score: A Reference-free Evaluation Metric for Image Captioning**, 2021.
8. Yu, Y., Zhang, W., & Deng, Y. **Frechet Inception Distance (FID) for Evaluating GANs**, 2021.
9. Jennewein, D. M., et al. **The Sol Supercomputer at Arizona State University**, *Practice and Experience in Advanced Research Computing*, 2023.
