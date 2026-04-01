import os
import sys
import shutil
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel

# --- Configuration ---
class InferenceConfig:
    root_dir = "./results/003_FirstFrameSD/"
    unet_path = "./results/003_FirstFrameSD/unet_final"
    
    # Path to your fine-tuned VAE from Step 1
    vae_path = "./results/001_vae_finetuned/"
    
    # Base model name (should match what you trained with)
    model_name = "runwayml/stable-diffusion-v1-5"
    
    # Output directory for generated images
    output_dir = "./results/003_FirstFrameSD/inference_results/"
    
    # Image generation settings
    img_size = (192,192)
    guidance_scale = 7.5  
    num_inference_steps = 50 
    seed = 42
    
    # System
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = InferenceConfig()

def main():
    os.makedirs(config.output_dir, exist_ok=True)
    
    # --- 1. Load the Fine-tuned Models ---
    print(f"Loading fine-tuned UNet from: {config.unet_path}")
    unet = UNet2DConditionModel.from_pretrained(config.unet_path)
    
    print(f"Loading fine-tuned VAE from: {config.vae_path}")
    vae = AutoencoderKL.from_pretrained(config.vae_path)

    text_encoder = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder")

    # --- 2. Create the Stable Diffusion Pipeline ---
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model_name,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        safety_checker=None, 
        feature_extractor=None
    ).to(config.device)
    
    print("Pipeline loaded successfully.")

    # --- 3. Define Prompts ---
    # These are the same prompts used for validation during training
    prompts = {
        "mid1": "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Normal function.",
        "mid2": "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Dilated Cardiomyopathy.",
        "mid3": "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Hypertrophic Cardiomyopathy.",
        "mid60kg": "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Normal function. Patient weight is 60 kg.",
        "mid70kg": "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Normal function. Patient weight is 70 kg.",
        "mid50kg": "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Normal function. Patient weight is 50 kg.",
        "hcm_apical": "A 2D cardiac MRI at the Apical level, shows End-Diastole for a patient with Hypertrophic Cardiomyopathy.",
        "mi_basal": "A 2D cardiac MRI at the Basal level, shows End-Diastole for a patient with Myocardial Infarction.",
        "dcm_mid": "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Dilated Cardiomyopathy."
    }

    # --- 4. Generate Images ---
    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    for name, prompt in prompts.items():
        print(f"\nGenerating image for: {name}")
        print(f"  Prompt: {prompt}")
        
        with torch.no_grad(), torch.autocast("cuda"):
            image = pipeline(
                prompt,
                height=config.img_size[0],
                width=config.img_size[1],
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            ).images[0]
        
        # Save the image
        output_path = os.path.join(config.output_dir, f"{name}_seed{config.seed}.png")
        image.save(output_path)
        print(f"  Image saved to: {output_path}")

if __name__ == "__main__":
    
    try:
        source_script_path = sys.argv[0]
        script_name = os.path.basename(source_script_path)
        dest_script_path = os.path.join(config.root_dir, script_name)
        shutil.copy(source_script_path, dest_script_path)
        print(f"Copied training script '{source_script_path}' to '{dest_script_path}'")
    except Exception as e:
        print(f"Warning: Could not copy training script. Error: {e}")
        
    main()