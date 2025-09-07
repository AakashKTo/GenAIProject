# -*- coding: utf-8 -*-
"""
This script provides a comprehensive framework for evaluating and analyzing Stable Diffusion models,
with a focus on benchmarking the "Faster Diffusion" optimization technique.

The script is divided into two main functional parts:
1.  **Benchmarking Framework:**
    -   Evaluates the performance and quality of Stable Diffusion models (v1.5 and v2).
    -   Compares different schedulers (DDIM, DPM-Solver, DPM-Solver-PP).
    -   Measures the impact of the "Faster Diffusion" optimization (`register_parallel_pipeline`
        and `register_faster_forward`).
    -   Calculates standard metrics: Generation Time, Fréchet Inception Distance (FID),
        and CLIP Score.
    -   Saves aggregated results to a CSV file for easy analysis.

2.  **Feature Extraction Framework:**
    -   Designed for deep model analysis of the UNet's internal workings.
    -   Uses PyTorch hooks to capture and save intermediate feature maps from the UNet's
      encoder (down_blocks) and decoder (up_blocks) at each diffusion timestep.
    -   Saves features to disk for later inspection, which is useful for research and
      understanding the image generation process.

To use this script, configure the settings in the `if __name__ == "__main__":` block
to select the desired mode (benchmarking or feature extraction) and model configurations.
"""

# ======================================================================================
# 1. IMPORTS & GLOBAL CONFIGURATION
# ======================================================================================
import sys
import torch
import pandas as pd
import os
import time
import gc
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Extend system path to include Faster Diffusion library directories
# Assumes the script is run from a directory containing these folders
sys.path.extend([
    "Faster-Diffusion"
])

# Diffusers and related libraries
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler
)
from torchvision.transforms import Compose, Resize
from torchmetrics.image.fid import FrechetInceptionDistance
import clip

# Import the Faster Diffusion optimization functions
from utils_sd import register_faster_forward, register_parallel_pipeline

# --- Global Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Path to the COCO prompts dataset file
PROMPTS_CSV_PATH = "coco_val2017_prompts.csv"
# Directory containing real images for FID calculation. This must be populated beforehand.
REAL_IMAGES_DIR = "real_images"
# Base directory for saving all generated images and results
BASE_RESULTS_DIR = "evaluation_outputs"

os.makedirs(BASE_RESULTS_DIR, exist_ok=True)


# ======================================================================================
# 2. SHARED UTILITY FUNCTIONS
# ======================================================================================

def get_pipeline(model_id, scheduler_type, timesteps, use_optimization=False):
    """
    Initializes and configures a Stable Diffusion pipeline.

    Args:
        model_id (str): The Hugging Face model identifier (e.g., "stabilityai/stable-diffusion-2").
        scheduler_type (str): The type of scheduler to use ("DDIM", "DPM-Solver", "DPM-Solver-PP").
        timesteps (int): The number of inference steps for the scheduler.
        use_optimization (bool): If True, applies the Faster Diffusion optimizations.

    Returns:
        diffusers.StableDiffusionPipeline: The configured pipeline object.
    """
    print(f"Loading model: {model_id} with {scheduler_type} scheduler...")

    # Select and configure the scheduler
    if scheduler_type == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_type == "DPM-Solver-PP":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_type == "DPM-Solver":
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # Load the main pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=TORCH_DTYPE
    ).to(DEVICE)

    # Set the number of inference steps
    pipe.scheduler.set_timesteps(timesteps)

    # Apply optimizations if requested
    if use_optimization:
        print("Applying Faster Diffusion optimizations...")
        register_parallel_pipeline(pipe) #
        register_faster_forward(pipe.unet) #

    return pipe


def compute_fid(fake_dir, real_dir):
    """
    Computes the Fréchet Inception Distance (FID) between two sets of images.

    Args:
        fake_dir (str): Path to the directory with generated images.
        real_dir (str): Path to the directory with real images.

    Returns:
        float: The calculated FID score, rounded to 4 decimal places.
    """
    print(f"Calculating FID between '{real_dir}' and '{fake_dir}'...")
    if not os.path.exists(real_dir) or not os.listdir(real_dir):
        print(f"Warning: Real images directory '{real_dir}' is empty or does not exist. FID cannot be computed.")
        return -1.0

    fid = FrechetInceptionDistance(feature=2048).to(DEVICE) #

    # Preprocessing transform for FID model (InceptionV3)
    transform = Compose([
        Resize((299, 299)), #
        lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).to(torch.uint8) #
    ])

    def update_fid_from_dir(img_dir, is_real):
        for fname in tqdm(os.listdir(img_dir), desc=f"Loading {'real' if is_real else 'fake'} images for FID"):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB") #
                img_tensor = transform(img).unsqueeze(0).to(DEVICE) #
                fid.update(img_tensor, real=is_real) #
            except Exception as e:
                print(f"Warning: Could not process image {fname} for FID. Error: {e}")

    # Update FID with real and generated images
    update_fid_from_dir(real_dir, is_real=True)
    update_fid_from_dir(fake_dir, is_real=False)

    return round(fid.compute().item(), 4) #


def calculate_clip_score(images_dir, prompts, clip_model, preprocess):
    """
    Calculates the average CLIP score between generated images and their prompts.

    Args:
        images_dir (str): Path to the directory with generated images.
        prompts (list of str): A list of prompts corresponding to the images.
        clip_model: The loaded CLIP model.
        preprocess: The CLIP image preprocessor.

    Returns:
        float: The average CLIP score, rounded to 4 decimal places.
    """
    print(f"Calculating CLIP score for images in '{images_dir}'...")
    scores = []
    
    # Create a mapping from prompt hash to prompt text for quick lookup
    prompt_map = {hashlib.md5(p.encode()).hexdigest(): p for p in prompts}

    for fname in tqdm(os.listdir(images_dir), desc="Calculating CLIP Scores"):
        if not fname.lower().endswith(".png"):
            continue

        # The image filename (without extension) is the hash of the prompt
        prompt_hash = os.path.splitext(fname)[0]
        prompt = prompt_map.get(prompt_hash)
        
        if prompt is None:
            continue
            
        img_path = os.path.join(images_dir, fname)
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE) #
            text = clip.tokenize([prompt]).to(DEVICE) #

            with torch.no_grad():
                im_feat = clip_model.encode_image(image) #
                txt_feat = clip_model.encode_text(text) #
            
            # Normalize features
            im_feat /= im_feat.norm(dim=-1, keepdim=True) #
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True) #

            # Calculate score (similarity * 2.5)
            score = 2.5 * torch.clamp((im_feat @ txt_feat.T).squeeze(), 0, 1).item() #
            scores.append(score)
        except Exception as e:
            print(f"Warning: Could not process image {fname} for CLIP Score. Error: {e}")

    if not scores:
        return 0.0
        
    return round(sum(scores) / len(scores), 4)


# ======================================================================================
# 3. BENCHMARKING FRAMEWORK
# ======================================================================================

def run_benchmarking_evaluation(configs, num_prompts, random_state=42):
    """
    Main function to run the benchmarking evaluation.

    Args:
        configs (list of dict): A list of configurations to test. Each dict should contain
                                'model_id', 'scheduler', 'steps', and 'optimized'.
        num_prompts (int): The number of prompts to use for the evaluation.
        random_state (int): A random seed for sampling prompts to ensure reproducibility.
    """
    print("--- Starting Benchmarking Evaluation ---")
    
    # Load prompts from dataset
    try:
        df = pd.read_csv(PROMPTS_CSV_PATH)
        if num_prompts > len(df):
            print(f"Warning: Requested {num_prompts} prompts, but dataset only has {len(df)}. Using all available prompts.")
            num_prompts = len(df)
        prompts = df.sample(n=num_prompts, random_state=random_state).reset_index(drop=True)["caption"].tolist() #
    except FileNotFoundError:
        print(f"Error: Prompts file not found at '{PROMPTS_CSV_PATH}'. Aborting.")
        return

    # Load CLIP model once to be reused for all evaluations
    print("Loading CLIP model for evaluations...")
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE) #

    all_results = []

    for config in configs:
        model_id = config['model_id']
        scheduler_type = config['scheduler']
        steps = config['steps']
        use_opt = config['optimized']
        
        method_name = f"{os.path.basename(model_id)}_{scheduler_type}_{steps}steps_{'Faster' if use_opt else 'Base'}"
        print(f"\n===== Evaluating: {method_name} =====")

        # Create a dedicated output directory for this configuration's images
        output_dir = os.path.join(BASE_RESULTS_DIR, method_name)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the pipeline for the current configuration
        pipe = get_pipeline(model_id, scheduler_type, steps, use_opt)
        
        # --- Image Generation and Performance Measurement ---
        generation_times = []
        for prompt in tqdm(prompts, desc=f"Generating images for {method_name}"):
            # Use CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            image = pipe(prompt, num_inference_steps=steps).images[0]
            end_event.record()
            torch.cuda.synchronize() # Wait for the generation to finish

            elapsed_time_ms = start_event.elapsed_time(end_event)
            generation_times.append(elapsed_time_ms / 1000.0) # Convert to seconds
            
            # Save image with a filename based on the hash of the prompt to avoid duplicates
            # and simplify mapping for CLIP score calculation.
            img_hash = hashlib.md5(prompt.encode()).hexdigest()
            img_path = os.path.join(output_dir, f"{img_hash}.png")
            image.save(img_path)

        # --- Metric Calculation ---
        avg_time = np.mean(generation_times) if generation_times else 0.0
        #fid_score = compute_fid(fake_dir=output_dir, real_dir=REAL_IMAGES_DIR)
        clip_score = calculate_clip_score(output_dir, prompts, clip_model, preprocess)

        # --- Store and Print Results ---
        result = {
            "Method": method_name,
            "Model": model_id,
            "Scheduler": scheduler_type,
            "Optimized": use_opt,
            "Steps": steps,
            #"FID↓": fid_score,
            "CLIP↑": clip_score,
            "Avg Time (s)↓": round(avg_time, 4)
        }
        all_results.append(result)
        
        print(f"----- Results for {method_name} -----")
        print(pd.DataFrame([result]).to_string(index=False))
        print("------------------------------------------")

        # Clean up memory before the next run
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

    # --- Final Results Compilation ---
    results_df = pd.DataFrame(all_results)
    print("\n\n===== FINAL CUMULATIVE RESULTS =====")
    print(results_df.to_string(index=False))
    
    # Save final results to a CSV file
    final_csv_path = os.path.join(BASE_RESULTS_DIR, "final_benchmark_results.csv")
    results_df.to_csv(final_csv_path, index=False)
    print(f"\nFinal results saved to '{final_csv_path}'")


# ======================================================================================
# 4. FEATURE EXTRACTION FRAMEWORK
# ======================================================================================

class FeatureRecorder:
    """
    A class to record intermediate features from a model's layers using hooks.
    
    Saves the encoder and decoder outputs for each step of the diffusion process.
    """
    def __init__(self, save_dir, prompt_hash):
        self.step_index = 0
        self.prompt_hash = prompt_hash
        self.save_dir = save_dir

    def record_hook(self, module, inp, out, layer_name):
        """The actual hook function that saves the output."""
        # Create directory for the current step if it doesn't exist
        step_dir = os.path.join(self.save_dir, self.prompt_hash, f"step_{self.step_index:03d}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Detach tensor from graph and move to CPU to save memory
        if isinstance(out, tuple):
            feature_to_save = tuple(o.detach().cpu() for o in out if torch.is_tensor(o))
        else:
            feature_to_save = out.detach().cpu()

        # Save the feature
        save_path = os.path.join(step_dir, f"{layer_name}.pt")
        torch.save(feature_to_save, save_path)


def run_feature_extraction(model_id, num_prompts, timesteps=50, random_state=42):
    """
    Main function to run the feature extraction process.

    This function generates images and, for each denoising step, saves the output
    of every down_block and up_block in the UNet.
    """
    print("--- Starting Feature Extraction ---")

    # Load prompts
    try:
        df = pd.read_csv(PROMPTS_CSV_PATH)
        prompts = df.sample(n=num_prompts, random_state=random_state).reset_index(drop=True)["caption"].tolist() #
    except FileNotFoundError:
        print(f"Error: Prompts file not found at '{PROMPTS_CSV_PATH}'. Aborting.")
        return

    # Create a pipeline (no optimizations needed for this task)
    pipe = get_pipeline(model_id, "DDIM", timesteps, use_optimization=False) #
    
    save_dir = os.path.join(BASE_RESULTS_DIR, "feature_outputs")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Features will be saved in: '{save_dir}'")

    for prompt in tqdm(prompts, desc="Processing prompts for feature extraction"):
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest() #
        
        # --- Register Hooks ---
        recorder = FeatureRecorder(save_dir, prompt_hash) #
        hooks = []
        for i, block in enumerate(pipe.unet.down_blocks):
            hook = block.register_forward_hook(
                lambda m, i, o, name=f"down_{i}": recorder.record_hook(m, i, o, name)
            )
            hooks.append(hook)
        for i, block in enumerate(pipe.unet.up_blocks):
            hook = block.register_forward_hook(
                lambda m, i, o, name=f"up_{i}": recorder.record_hook(m, i, o, name)
            )
            hooks.append(hook)

        # --- Generation Loop (Manual) ---
        # This is a simplified version of the pipeline's __call__ method
        # to allow intervention at each step.
        
        # 1. Get text embeddings
        text_inputs = pipe.tokenizer(
            prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
        )
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(DEVICE))[0] #
        
        # 2. Prepare initial latents
        latents = torch.randn((1, 4, 96, 96), device=DEVICE, dtype=TORCH_DTYPE) #

        # 3. Denoising loop
        for i, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="Denoising steps")):
            recorder.step_index = i # Update step index for recorder
            
            with torch.no_grad():
                # The forward pass through the UNet will trigger the hooks
                noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_embeddings).sample #
                # Update latents for the next step
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample #
        
        # --- Cleanup ---
        # Remove hooks to prevent them from firing on subsequent runs
        for hook in hooks:
            hook.remove()
        
        del recorder
        torch.cuda.empty_cache()
        gc.collect()


# ======================================================================================
# 5. SCRIPT EXECUTION
# ======================================================================================
if __name__ == "__main__":

    # --- CHOOSE MODE ---
    # Set to 'benchmark' to run performance/quality tests.
    # Set to 'features' to run the feature extraction.
    MODE = "benchmark"

    if MODE == "benchmark":
        # --- Configuration for Benchmarking ---
        # Define the list of experiments you want to run.
        # Add or remove dictionaries to test different models and settings.
        BENCHMARK_CONFIGS = [
            # --- Stable Diffusion v2 ---
            {"model_id": "stabilityai/stable-diffusion-2", "scheduler": "DPM-Solver-PP", "steps": 20, "optimized": False},
            {"model_id": "stabilityai/stable-diffusion-2", "scheduler": "DPM-Solver-PP", "steps": 20, "optimized": True},
            {"model_id": "stabilityai/stable-diffusion-2", "scheduler": "DDIM", "steps": 50, "optimized": False},
            {"model_id": "stabilityai/stable-diffusion-2", "scheduler": "DDIM", "steps": 50, "optimized": True},

            # --- Stable Diffusion v1.5 ---
            {"model_id": "runwayml/stable-diffusion-v1-5", "scheduler": "DPM-Solver-PP", "steps": 20, "optimized": False},
            {"model_id": "runwayml/stable-diffusion-v1-5", "scheduler": "DPM-Solver-PP", "steps": 20, "optimized": True},
            {"model_id": "runwayml/stable-diffusion-v1-5", "scheduler": "DDIM", "steps": 20, "optimized": False},
            {"model_id": "runwayml/stable-diffusion-v1-5", "scheduler": "DDIM", "steps": 20, "optimized": True},
        ]
        NUM_BENCHMARK_PROMPTS = 1 # Adjust for quicker or more thorough testing

        run_benchmarking_evaluation(
            configs=BENCHMARK_CONFIGS,
            num_prompts=NUM_BENCHMARK_PROMPTS
        )

    elif MODE == "features":
        # --- Configuration for Feature Extraction ---
        FEATURE_MODEL_ID = "stabilityai/stable-diffusion-2-1" #
        NUM_FEATURE_PROMPTS = 5 # Keep this low as it generates a lot of data
        
        run_feature_extraction(
            model_id=FEATURE_MODEL_ID,
            num_prompts=NUM_FEATURE_PROMPTS
        )
        
    else:
        print(f"Error: Unknown MODE '{MODE}'. Please choose 'benchmark' or 'features'.")