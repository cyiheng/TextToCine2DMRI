import os
import sys
import shutil
import glob
import re
from tqdm import tqdm
import matplotlib.cm as cm
import imageio # Used for creating GIFs
import numpy as np

import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL
from monai import transforms as mt

from utils import warp_latent, flow_to_rgb
from models.blocks import LatentFlowAndResidualPredictor

# --- Configuration ---
class Config:
    dataset_root = "./data/dataset/"
    
    train_output_dir = "./results/002_lfm/"
    
    # Construct model paths from the main directory
    vae_model_path = "./results/001_vae_finetuned/"
    flow_model_checkpoint = os.path.join(train_output_dir, "checkpoints/best_flow_predictor.pth")
    
    output_dir = os.path.join(train_output_dir, "inference_results/")
    
    # --- SELECT PATIENT AND SLICE TO TEST ---
    patient_slice_to_test = "ACDC_patient145_slice1" 
    
    # Model and data parameters
    img_size = (192, 192)
    latent_channels = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()


# --- Main Inference Function ---
@torch.no_grad()
def generate_sequence():
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Load models
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(config.vae_model_path).to(config.device).eval()
    
    flow_predictor = LatentFlowAndResidualPredictor(
        in_channels=2 * config.latent_channels, 
        latent_channels=config.latent_channels
    ).to(config.device)
    flow_predictor.load_state_dict(torch.load(config.flow_model_checkpoint, map_location=config.device))
    flow_predictor.eval()
    
    print("Models loaded successfully.")
    
    # 2. Prepare data sequence
    slice_path = os.path.join(config.dataset_root, config.patient_slice_to_test)
    frame_paths = sorted(glob.glob(os.path.join(slice_path, "*.png")), key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
    print(f"Found {len(frame_paths)} frames for {config.patient_slice_to_test}.")
    transforms = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True), mt.RepeatChannel(repeats=3),
        mt.ResizeWithPadOrCrop(spatial_size=config.img_size, mode="constant", constant_values=0),
        mt.ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
    ])
    gt_frames = [transforms(p).to(config.device) for p in frame_paths]

    
    # 3. Autoregressive generation loop
    print("Generating sequence...")
    # Initialize with the latent of the first ground truth frame
    z_current = vae.encode(gt_frames[0].unsqueeze(0)).latent_dist.sample()
    
    generated_frames = [gt_frames[0]]
    viz_frames = []

    for i in tqdm(range(len(gt_frames)), desc="Generating frames"):
        # The first frame is just the ground truth, we start "generating" from the second
        if i == 0:
            # For the first frame, we create placeholder visualizations (all black)
            new_frame_tensor = gt_frames[0]
            placeholder_viz = np.zeros((config.img_size[0], config.img_size[1], 3), dtype=np.uint8)
            viz_data = {
                "flow_x": placeholder_viz, "flow_y": placeholder_viz, 
                "flow_rgb": placeholder_viz, "residual": placeholder_viz
            }
        else:
            # --- The "Driving" latent comes from the NEXT GT frame ---
            # This is "GT-driven" generation, not fully autoregressive.
            z_driving = vae.encode(gt_frames[i].unsqueeze(0)).latent_dist.sample()

            # --- Forward Pass ---
            latent_flow, latent_residual = flow_predictor(z_current, z_driving)
            z_warped = warp_latent(z_current, latent_flow)
            z_pred = z_warped + latent_residual
            # Decode the final latent to get the new image
            new_frame_tensor = vae.decode(z_pred).sample

            # Upsample the latent-space flow/residual to image size for visualization
            flow_img_space = F.interpolate(latent_flow, size=config.img_size, mode='bilinear', align_corners=False)
            res_img_space = F.interpolate(latent_residual, size=config.img_size, mode='bilinear', align_corners=False)
            
            flow_x_np = flow_img_space[0, 0].cpu().numpy()
            flow_y_np = flow_img_space[0, 1].cpu().numpy()
            res_0_np = res_img_space[0, 0].cpu().numpy()
            
            # Normalize for coolwarm colormap
            vmax_flow = np.abs(flow_img_space.cpu().numpy()).max() + 1e-6
            vmax_res = np.abs(res_0_np).max() + 1e-6

            viz_data = {
                "flow_x": (cm.coolwarm( (flow_x_np + vmax_flow) / (2 * vmax_flow) )[:,:,:3] * 255).astype(np.uint8),
                "flow_y": (cm.coolwarm( (flow_y_np + vmax_flow) / (2 * vmax_flow) )[:,:,:3] * 255).astype(np.uint8),
                "flow_rgb": flow_to_rgb(flow_img_space),
                "residual": (cm.coolwarm( (res_0_np + vmax_res) / (2 * vmax_res) )[:,:,:3] * 255).astype(np.uint8),
            }

        generated_frames.append(new_frame_tensor.squeeze(0))
        viz_frames.append(viz_data)

    # 4. Save results as a GIF with enhanced visualizations
    print("Saving results as a GIF...")
    

    def process_for_gif(tensor):
        img = (tensor.clamp(-1, 1) * 0.5 + 0.5)
        img = img.permute(1, 2, 0).cpu().numpy()
        return (img * 255).astype('uint8')

    comparison_frames = []
    for i in range(len(gt_frames)):
        gt_img = process_for_gif(gt_frames[i])
        gen_img = process_for_gif(generated_frames[i])
        
        # Calculate difference map
        diff = np.abs(gt_img[:,:,0].astype(np.float32) - gen_img[:,:,0].astype(np.float32))
        diff_norm = diff / 255.0
        diff_colored = (cm.coolwarm(diff_norm)[:,:,:3] * 255).astype('uint8')

        # Get the visualization frames for this step
        viz_data = viz_frames[i]
        
        combined_frame = np.concatenate((
            gt_img, 
            gen_img, 
            diff_colored,
            viz_data["flow_x"],
            viz_data["flow_y"],
            viz_data["flow_rgb"],
            viz_data["residual"]
        ), axis=1)
        comparison_frames.append(combined_frame)
    
    gif_path = os.path.join(config.output_dir, f"inference_viz_{config.patient_slice_to_test}.gif")
    imageio.mimsave(gif_path, comparison_frames, fps=10, loop=0)
    
    print(f"Successfully saved inference GIF to: {gif_path}")
    print("Layout: GT | Generated | Diff | Flow-X | Flow-Y | Flow-RGB | Residual-Ch0")



if __name__ == "__main__":
    try:
        source_script_path = sys.argv[0]
        script_name = os.path.basename(source_script_path)
        dest_script_path = os.path.join(config.train_output_dir, script_name)
        if not os.path.exists(dest_script_path):
             shutil.copy(source_script_path, dest_script_path)
             print(f"Copied inference script to '{dest_script_path}'")
    except Exception as e:
        print(f"Warning: Could not copy inference script. Error: {e}")

    generate_sequence()