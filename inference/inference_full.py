import os
import torch
import torch.nn as nn
import numpy as np
import imageio
from tqdm.auto import tqdm
import math
import matplotlib.pyplot as plt
from safetensors.torch import load_file

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from monai import transforms as mt
from utils import warp_latent
from models.blocks import SinusoidalPositionalEmbedding

# --- Main Configuration ---
class InferenceConfig:
    finetuned_vae_path = "./results/001_vae_finetuned/"
    first_frame_unet_path = "./results/003_FirstFrameSD/unet_final/"
    flow_unet_path = "./results/004_FlowSD/unet_final/"

    model_name = "runwayml/stable-diffusion-v1-5"

    # --- Inference Parameters ---
    prompt = "A 2D cardiac MRI at the basal level, shows End-Diastole for a patient with Abnormal Right ventricular function."
    output_dir = "./results/final_inference_output"
    
    num_frames_to_generate = 20
    num_inference_steps_frame1 = 30 # Steps for the first frame
    num_inference_steps_motion = 50 # Steps for the flow generation
    guidance_scale = 7.5 # For classifier-free guidance on the first frame
    seed = 42
    
    img_size = (192,192)
    
    # Model architecture parameters (must match training)
    vae_latent_channels = 4
    flow_channels = 2
    transformation_channels = 2 + vae_latent_channels # Total 6 channels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_cardiac_time_signal(num_frames: int) -> torch.Tensor:
    """
    Generates a non-linear time signal that mimics a cardiac cycle.
    
    The signal uses a sinusoidal ease-in-out curve for both the
    systole (0 -> 100) and diastole (100 -> 200) phases.
    
    Args:
        num_frames: The total number of frames in the video.
        
    Returns:
        A torch.Tensor of shape (num_frames,) with the time values.
    """
    print(f"Generating non-linear time signal for {num_frames} frames.")
    
    # Split frames into contraction (systole) and relaxation (diastole)
    # We'll give systole roughly 40% of the frames and diastole 60%
    num_systole_frames = max(2, int(num_frames * 0.4))
    num_diastole_frames = num_frames - num_systole_frames

    # Phase 1: Systole (ED to ES, time 0 to 100)
    # We use a cosine curve to create an ease-in-out effect
    systole_progress = np.linspace(0, np.pi, num_systole_frames)
    systole_easing = 0.5 * (1 - np.cos(systole_progress)) # Ranges from 0 to 1
    systole_times = 100 * systole_easing
    
    # Phase 2: Diastole (ES to ED, time 100 to 200)
    diastole_progress = np.linspace(0, np.pi, num_diastole_frames)
    diastole_easing = 0.5 * (1 - np.cos(diastole_progress)) # Ranges from 0 to 1
    diastole_times = 100 + (100 * diastole_easing)
    
    # Combine the two phases
    time_signal = np.concatenate([systole_times, diastole_times])
    
    # Ensure the final signal has the correct number of frames due to rounding
    # and that the last value is exactly 200.
    final_signal = np.interp(
        np.linspace(0, 1, num_frames),
        np.linspace(0, 1, len(time_signal)),
        time_signal
    )
    final_signal[-1] = 200.0
    
    return torch.from_numpy(final_signal).float()

def generate_text_to_cine(config):
    """
    Main function to generate a full cine MRI sequence from a text prompt.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    # --- 1. Load All Necessary Models ---
    print("Loading all pre-trained models...")
    
    # --- Models for Stage 1 (First Frame Generation) ---
    vae = AutoencoderKL.from_pretrained(config.finetuned_vae_path, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder", local_files_only=True)
    first_frame_unet = UNet2DConditionModel.from_pretrained(config.first_frame_unet_path)
    
    # --- Models for Stage 2 (Motion Generation) ---
    # Load the base UNet architecture for flow generation
    flow_unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet")
    
    # Manually modify the flow UNet's layers before loading weights
    new_in_channels = config.transformation_channels + config.vae_latent_channels # 6 + 4 = 10
    flow_unet.conv_in = nn.Conv2d(new_in_channels, flow_unet.conv_in.out_channels, kernel_size=3, padding=1)
    flow_unet.conv_out = nn.Conv2d(flow_unet.conv_out.in_channels, config.transformation_channels, kernel_size=3, padding=1)
    
    # Load fine-tuned weights from the .safetensors file
    flow_weights_path = os.path.join(config.flow_unet_path, "diffusion_pytorch_model.safetensors")
    flow_unet.load_state_dict(load_file(flow_weights_path, device="cpu"))
    
    time_embedder = SinusoidalPositionalEmbedding(flow_unet.config.cross_attention_dim)
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")

    # --- STAGE 1: GENERATE THE FIRST FRAME FROM TEXT ---
    print("\n--- Stage 1: Generating the first frame from prompt ---")
    
    pipeline = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=first_frame_unet,
        scheduler=noise_scheduler, safety_checker=None, feature_extractor=None,
    ).to(config.device)
    
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    
    with torch.autocast("cuda"):
        result = pipeline(
            prompt=config.prompt,
            num_inference_steps=config.num_inference_steps_frame1,
            guidance_scale=config.guidance_scale,
            generator=generator,
            height=config.img_size[0],
            width=config.img_size[1]
        )
    reference_image_pil = result.images[0]
    output_path = os.path.join(config.output_dir, f"firstframe.png")
    reference_image_pil.save(output_path)
    
    print("First frame generated successfully.")

    # --- 2. PREPARE REFERENCE TENSOR FOR STAGE 2 USING THE CORRECT MONAI PIPELINE ---
    print("Preprocessing generated frame with MONAI pipeline for consistency...")

    # Define a MONAI transform pipeline for in-memory NumPy arrays.
    # This pipeline *must match* the one used in Step 1 (VAE fine-tuning).
    # It takes a grayscale NumPy array as input.
    in_memory_transforms = mt.Compose([
        mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"), # <-- THE FIX IS HERE
        mt.RepeatChanneld(keys=["image"], repeats=3),
        mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=config.img_size, mode="constant", constant_values=0),
        mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
    ])

    # Convert the generated PIL image to a grayscale NumPy array
    reference_image_pil_gray = reference_image_pil.convert("L")
    ref_np_gray = np.array(reference_image_pil_gray)

    # Apply the MONAI transforms
    ref_dict = in_memory_transforms({"image": ref_np_gray})
    reference_image_tensor = ref_dict["image"].unsqueeze(0).to(config.device)
    
    # --- STAGE 2: GENERATE MOTION SEQUENCE ---
    print("\n--- Stage 2: Generating motion sequence ---")
    
    # Move necessary models to device and set to eval mode
    vae.to(config.device).eval()
    flow_unet.to(config.device).eval()
    time_embedder.to(config.device).eval()

    generated_frames = []
    generated_flow_magnitudes = []
    generated_residual_magnitudes = []
    generated_frames.append(ref_np_gray)
    # --- END OF FIX 1 ---
    
    # Encode the reference image once
    with torch.no_grad():
        ref_latent = vae.encode(reference_image_tensor).latent_dist.sample()
        input_ids = tokenizer(config.prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(config.device)
        text_embeddings = text_encoder(input_ids)[0]

    
    latent_h, latent_w = ref_latent.shape[2], ref_latent.shape[3]
    blank_latent_viz = np.zeros((latent_h, latent_w))
    generated_flow_magnitudes.append(blank_latent_viz)
    generated_residual_magnitudes.append(blank_latent_viz)

    time_steps = generate_cardiac_time_signal(config.num_frames_to_generate)
    
    # We skip the first time step (t=0) as it's the reference frame
    for i in tqdm(range(1, len(time_steps)), desc="Generating subsequent frames"):
        time_val = time_steps[i]
        
        latent_transformation = torch.randn(
            (1, config.transformation_channels, ref_latent.shape[2], ref_latent.shape[3]),
            generator=generator, device=config.device
        )

        noise_scheduler.set_timesteps(config.num_inference_steps_motion)
        
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                cardiac_time_embedding = time_embedder(torch.tensor([time_val], device=config.device)).unsqueeze(1)
                cond_emb = torch.cat([text_embeddings, cardiac_time_embedding], dim=1)
                unet_input = torch.cat([latent_transformation , ref_latent], dim=1)
                noise_pred = flow_unet(unet_input, t, encoder_hidden_states=cond_emb).sample
                latent_transformation  = noise_scheduler.step(noise_pred, t, latent_transformation).prev_sample

        with torch.no_grad():
            generated_flow = latent_transformation[:, :2, :, :]
            generated_residual = latent_transformation[:, 2:, :, :]

            flow_magnitude = torch.sqrt(torch.sum(generated_flow**2, dim=1)).squeeze().cpu().numpy()
            generated_flow_magnitudes.append(flow_magnitude)

            # Residual magnitude (L2 norm across the 4 channels)
            residual_magnitude = torch.linalg.norm(generated_residual, dim=1).squeeze().cpu().numpy()
            generated_residual_magnitudes.append(residual_magnitude)

            warped_latent_norm = warp_latent(ref_latent, generated_flow) + generated_residual
            image_tensor = vae.decode(warped_latent_norm).sample
            gray_tensor = image_tensor[0, 0, :, :]
            gray_tensor_0_1 = (gray_tensor / 2 + 0.5).clamp(0, 1)
            image_np_gray = (gray_tensor_0_1.cpu().numpy() * 255).astype(np.uint8)
            
            generated_frames.append(image_np_gray)

    # --- 4. Save Outputs ---
    print("\nSaving outputs...")
    
    # Save Grid Plot
    grid_size = math.ceil(math.sqrt(len(generated_frames)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axs = axs.flatten()
    for i, frame in enumerate(generated_frames):
        axs[i].imshow(frame, cmap="gray")
        axs[i].set_title(f"F {i}")
        axs[i].axis("off")
    for i in range(len(generated_frames), len(axs)):
        axs[i].axis("off")
    
    plt.tight_layout()
    plot_path = os.path.join(config.output_dir, "final_cine_grid.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved grid plot to: {plot_path}")

    grid_size = math.ceil(math.sqrt(len(generated_flow_magnitudes)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2))
    fig.suptitle("Predicted Flow Magnitudes", fontsize=16)
    axs = axs.flatten()
    for i, frame in enumerate(generated_flow_magnitudes):
        ax = axs[i]
        im = ax.imshow(frame, cmap="coolwarm")
        ax.set_title(f"F {i} -> F {i+1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(generated_flow_magnitudes), len(axs)):
        axs[i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(config.output_dir, "final_flow_magnitude_grid.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved flow magnitude grid plot to: {plot_path}")


    # --- Save Residual Magnitude Grid Plot ---
    grid_size = math.ceil(math.sqrt(len(generated_residual_magnitudes)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2))
    fig.suptitle("Predicted Residual Magnitudes", fontsize=16)
    axs = axs.flatten()
    for i, frame in enumerate(generated_residual_magnitudes):
        ax = axs[i]
        im = ax.imshow(frame, cmap="hot")
        ax.set_title(f"F {i} -> F {i+1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(generated_residual_magnitudes), len(axs)):
        axs[i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(config.output_dir, "final_residual_magnitude_grid.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved residual magnitude grid plot to: {plot_path}")

    # Save Animated GIF
    gif_path = os.path.join(config.output_dir, "final_cine_motion.gif")
    imageio.mimsave(gif_path, generated_frames, fps=10, loop=0)
    print(f"Saved animated GIF to: {gif_path}")

if __name__ == "__main__":
    config = InferenceConfig()
    generate_text_to_cine(config)