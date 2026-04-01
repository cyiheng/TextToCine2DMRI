import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio
from tqdm.auto import tqdm
import math
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import torch.nn.functional as F
import gradio as gr
import tempfile
import time

# --- Import necessary components ---
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from monai import transforms as mt
from utils import warp_latent
from models.blocks import SinusoidalPositionalEmbedding

# --- Main Configuration ---
class InferenceConfig:
    # --- REQUIRED: Paths to your trained models ---
    # Please ensure these paths are correct for your system
    finetuned_vae_path = "./results/001_vae_finetuned/"
    first_frame_unet_path = "./results/003_FirstFrameSD/unet_final/"
    flow_unet_path = "./results/004_FlowSD/unet_final/"
    model_name = "runwayml/stable-diffusion-v1-5"
    img_size = (192, 192)
    vae_latent_channels = 4
    flow_channels = 2
    transformation_channels = 2 + vae_latent_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_cardiac_time_signal(num_frames: int) -> torch.Tensor:
    num_systole_frames = max(2, int(num_frames * 0.4))
    num_diastole_frames = num_frames - num_systole_frames
    systole_progress = np.linspace(0, np.pi, num_systole_frames)
    systole_easing = 0.5 * (1 - np.cos(systole_progress))
    systole_times = 100 * systole_easing
    diastole_progress = np.linspace(0, np.pi, num_diastole_frames)
    diastole_easing = 0.5 * (1 - np.cos(diastole_progress))
    diastole_times = 100 + (100 * diastole_easing)
    time_signal = np.concatenate([systole_times, diastole_times])
    final_signal = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(time_signal)), time_signal)
    final_signal[-1] = 200.0
    return torch.from_numpy(final_signal).float()

# --- Global Model Storage ---
MODELS = {}

def load_models():
    """Loads all models into a global dictionary to avoid reloading."""
    if MODELS: # If models are already loaded, do nothing
        print("Models are already loaded.")
        return

    print("Loading all pre-trained models... This may take a moment.")
    config = InferenceConfig()

    # Stage 1 Models
    MODELS['vae'] = AutoencoderKL.from_pretrained(config.finetuned_vae_path, local_files_only=True)
    MODELS['tokenizer'] = CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer", local_files_only=True)
    MODELS['text_encoder'] = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder", local_files_only=True)
    MODELS['first_frame_unet'] = UNet2DConditionModel.from_pretrained(config.first_frame_unet_path)

    # Stage 2 Models
    flow_unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet")
    new_in_channels = config.transformation_channels + config.vae_latent_channels
    flow_unet.conv_in = nn.Conv2d(new_in_channels, flow_unet.conv_in.out_channels, kernel_size=3, padding=1)
    flow_unet.conv_out = nn.Conv2d(flow_unet.conv_out.in_channels, config.transformation_channels, kernel_size=3, padding=1)
    flow_weights_path = os.path.join(config.flow_unet_path, "diffusion_pytorch_model.safetensors")
    flow_unet.load_state_dict(load_file(flow_weights_path, device="cpu"))
    MODELS['flow_unet'] = flow_unet
    
    MODELS['time_embedder'] = SinusoidalPositionalEmbedding(flow_unet.config.cross_attention_dim)
    MODELS['noise_scheduler'] = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")
    
    print("All models loaded successfully.")


def generate_text_to_cine(prompt, num_frames, guidance_scale, seed, steps_frame1, steps_motion, progress=gr.Progress(track_tqdm=True)):
    """
    Main generation function for the Gradio interface.
    """
    config = InferenceConfig()

    # Use a temporary directory for all outputs
    output_dir = tempfile.mkdtemp()

    # Handle random seed
    if seed == -1:
        seed = np.random.randint(0, 1_000_000)
    
    generator = torch.Generator(device=config.device).manual_seed(seed)
    
    # --- STAGE 1: GENERATE THE FIRST FRAME FROM TEXT ---
    print("\n--- Stage 1: Generating the first frame from prompt ---")
    
    pipeline = StableDiffusionPipeline(
        vae=MODELS['vae'], text_encoder=MODELS['text_encoder'], tokenizer=MODELS['tokenizer'], unet=MODELS['first_frame_unet'],
        scheduler=MODELS['noise_scheduler'], safety_checker=None, feature_extractor=None,
    ).to(config.device)
    
    with torch.autocast("cuda"):
        result = pipeline(
            prompt=prompt, num_inference_steps=steps_frame1, guidance_scale=guidance_scale,
            generator=generator, height=config.img_size[0], width=config.img_size[1]
        )
    reference_image_pil = result.images[0]
    first_frame_path = os.path.join(output_dir, "firstframe.png")
    reference_image_pil.save(first_frame_path)
    
    print("First frame generated successfully.")

    # --- PREPARE REFERENCE TENSOR FOR STAGE 2 ---
    in_memory_transforms = mt.Compose([
        mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        mt.RepeatChanneld(keys=["image"], repeats=3),
        mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=config.img_size, mode="constant", constant_values=0),
        mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
    ])
    reference_image_pil_gray = reference_image_pil.convert("L")
    ref_np_gray = np.array(reference_image_pil_gray)
    ref_dict = in_memory_transforms({"image": ref_np_gray})
    reference_image_tensor = ref_dict["image"].unsqueeze(0).to(config.device)
    
    # --- STAGE 2: GENERATE MOTION SEQUENCE ---
    print("\n--- Stage 2: Generating motion sequence ---")
    
    vae = MODELS['vae'].to(config.device).eval()
    flow_unet = MODELS['flow_unet'].to(config.device).eval()
    time_embedder = MODELS['time_embedder'].to(config.device).eval()
    text_encoder = MODELS['text_encoder'].to(config.device).eval()
    tokenizer = MODELS['tokenizer']
    noise_scheduler = MODELS['noise_scheduler']

    generated_frames = [ref_np_gray]
    generated_flow_magnitudes = []
    generated_residual_magnitudes = []
    
    with torch.no_grad():
        ref_latent = vae.encode(reference_image_tensor).latent_dist.sample()
        input_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(config.device)
        text_embeddings = text_encoder(input_ids)[0]

    latent_h, latent_w = ref_latent.shape[2], ref_latent.shape[3]
    generated_flow_magnitudes.append(np.zeros((latent_h, latent_w)))
    generated_residual_magnitudes.append(np.zeros((latent_h, latent_w)))

    time_steps = generate_cardiac_time_signal(num_frames)
    
    for i in tqdm(range(1, len(time_steps)), desc="Generating subsequent frames"):
        time_val = time_steps[i]
        
        latent_transformation = torch.randn(
            (1, config.transformation_channels, latent_h, latent_w),
            generator=generator, device=config.device
        )
        noise_scheduler.set_timesteps(steps_motion)
        
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                cardiac_time_embedding = time_embedder(torch.tensor([time_val], device=config.device)).unsqueeze(1)
                cond_emb = torch.cat([text_embeddings, cardiac_time_embedding], dim=1)
                unet_input = torch.cat([latent_transformation, ref_latent], dim=1)
                noise_pred = flow_unet(unet_input, t, encoder_hidden_states=cond_emb).sample
                latent_transformation = noise_scheduler.step(noise_pred, t, latent_transformation).prev_sample

        with torch.no_grad():
            generated_flow = latent_transformation[:, :2, :, :]
            generated_residual = latent_transformation[:, 2:, :, :]

            flow_magnitude = torch.sqrt(torch.sum(generated_flow**2, dim=1)).squeeze().cpu().numpy()
            generated_flow_magnitudes.append(flow_magnitude)

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
        axs[i].imshow(frame, cmap="gray"); axs[i].set_title(f"F {i}"); axs[i].axis("off")
    for i in range(len(generated_frames), len(axs)): axs[i].axis("off")
    plt.tight_layout()
    frames_grid_path = os.path.join(output_dir, "final_cine_grid.png")
    plt.savefig(frames_grid_path); plt.close()

    # Save Flow Grid Plot
    grid_size = math.ceil(math.sqrt(len(generated_flow_magnitudes)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2))
    fig.suptitle("Predicted Flow Magnitudes", fontsize=16); axs = axs.flatten()
    for i, frame in enumerate(generated_flow_magnitudes):
        ax = axs[i]; im = ax.imshow(frame, cmap="coolwarm"); ax.set_title(f"F {i} -> F {i+1}"); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(generated_flow_magnitudes), len(axs)): axs[i].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    flow_grid_path = os.path.join(output_dir, "final_flow_magnitude_grid.png")
    plt.savefig(flow_grid_path); plt.close()

    # Save Residual Grid Plot
    grid_size = math.ceil(math.sqrt(len(generated_residual_magnitudes)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2))
    fig.suptitle("Predicted Residual Magnitudes", fontsize=16); axs = axs.flatten()
    for i, frame in enumerate(generated_residual_magnitudes):
        ax = axs[i]; im = ax.imshow(frame, cmap="hot"); ax.set_title(f"F {i} -> F {i+1}"); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(generated_residual_magnitudes), len(axs)): axs[i].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    residual_grid_path = os.path.join(output_dir, "final_residual_magnitude_grid.png")
    plt.savefig(residual_grid_path); plt.close()

    # Save Animated GIF
    gif_path = os.path.join(output_dir, "final_cine_motion.gif")
    imageio.mimsave(gif_path, generated_frames, fps=10, loop=0)
    print(f"Generation complete. Outputs saved in {output_dir}")

    return first_frame_path, frames_grid_path, flow_grid_path, residual_grid_path, gif_path


# --- Build Gradio UI ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Text-to-Cine MRI Generation")
        gr.Markdown(
            "This application generates a cardiac cine MRI sequence from a text prompt. "
            "It first generates a starting frame (End-Diastole) and then predicts the motion for the subsequent frames "
            "based on the prompt and a non-linear cardiac time signal."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Medical Prompt",
                    value="A 2D cardiac MRI at the basal level, shows End-Diastole for a patient with Abnormal Right ventricular function.",
                    lines=3
                )
                num_frames = gr.Slider(
                    label="Number of Frames to Generate",
                    minimum=5,
                    maximum=40,
                    value=20,
                    step=1
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale (for first frame)",
                    minimum=1.0,
                    maximum=15.0,
                    value=7.5,
                    step=0.5
                )
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    info="Use -1 for a random seed."
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    steps_frame1 = gr.Slider(
                        label="Inference Steps (First Frame)",
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=1
                    )
                    steps_motion = gr.Slider(
                        label="Inference Steps (Motion)",
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=1
                    )

                generate_btn = gr.Button("Generate Cine Sequence", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Generated Cine (GIF)"):
                        output_gif = gr.Image(label="Cine MRI Sequence", type="filepath")
                    with gr.TabItem("Frame-by-Frame Grid"):
                        output_frames_grid = gr.Image(label="All Generated Frames", type="filepath")
                    with gr.TabItem("Motion Analysis"):
                        with gr.Row():
                            output_flow_grid = gr.Image(label="Latent Flow Magnitude", type="filepath")
                            output_residual_grid = gr.Image(label="Latent Residual Magnitude", type="filepath")
                    with gr.TabItem("Initial Frame"):
                         output_first_frame = gr.Image(label="Generated First Frame", type="filepath")

        inputs = [prompt, num_frames, guidance_scale, seed, steps_frame1, steps_motion]
        outputs = [output_first_frame, output_frames_grid, output_flow_grid, output_residual_grid, output_gif]
        
        generate_btn.click(fn=generate_text_to_cine, inputs=inputs, outputs=outputs)

    return demo

if __name__ == "__main__":
    # Load models once at startup
    load_models()
    
    # Create and launch the Gradio app
    app = create_ui()
    app.launch()