import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import re
import imageio
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import SimpleITK as sitk
import glob

# --- Import necessary components ---
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from monai import transforms as mt
from utils import  warp_latent
from models.blocks import SinusoidalPositionalEmbedding

DIAGNOSIS_GROUP_MAPPING = {"NOR": "Normal function", "MINF": "Myocardial Infarction", "DCM": "Dilated Cardiomyopathy", "HCM": "Hypertrophic Cardiomyopathy", "RV": "Abnormal Right Ventricle"}
def format_patient_id_key(path_string):
    match = re.search(r'patient(\d+)', path_string)
    if match: return f"patient{int(match.group(1)):03d}"
    return None

# --- Configuration ---
class InferenceConfig:
    base_data_dir = "./data/"
    train_output_dir = "./results/004_FlowSD/"
    flow_unet_path = os.path.join(train_output_dir, "unet_final/")
    finetuned_vae_path = "./results/001_vae_finetuned"
    anatomical_json_path = "./data/anatomical_levels.json"
    model_name = "runwayml/stable-diffusion-v1-5"
    dataset_root = "./data/dataset/"
    reference_image_path = os.path.join(dataset_root, "ACDC_patient101_slice1", "frame_000.png")
    output_dir = os.path.join(train_output_dir, "inference_output")
    img_size = (192, 192)
    num_inference_steps = 50
    seed = 42
    vae_latent_channels = 4
    transformation_channels = 2 + vae_latent_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_patient_metadata(base_data_dir):
    all_metadata = {}
    datasets_config = {
        'ACDC': {'folder': os.path.join(base_data_dir, 'ACDC_Preprocessed'), 'id_col': 'pid', 'info_cols': ["pathology", "height", "weight", "ed_frame", "es_frame"]},
        'DSB': {'folder': os.path.join(base_data_dir, 'DSB_nifti'), 'id_col': 'pid', 'info_cols': ['Sex', 'Age', 'es_frame']}
    }
    for dataset_name, params in datasets_config.items():
        for split in ['train', 'test', 'val']:
            csv_path = os.path.join(params['folder'], f"{split}_metadata.csv")
            if dataset_name == "DSB": csv_path = os.path.join(params['folder'], f"{split}_metadata_additional_with_es.csv")
            if not os.path.exists(csv_path): continue
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    patient_id_val = row[params['id_col']]
                    match = re.search(r"(\d+)", str(patient_id_val))
                    if not match: continue
                    patient_num = int(match.group(1))
                    unique_patient_key = (dataset_name, patient_num)
                    all_metadata[unique_patient_key] = {col: row.get(col) for col in params['info_cols']}
            except Exception as e:
                print(f"  [Error] Failed to process {csv_path}: {e}")
    return all_metadata

def build_inference_prompt(reference_image_path, cfg):
    all_metadata = load_all_patient_metadata(cfg.base_data_dir)
    with open(cfg.anatomical_json_path, 'r') as f:
        anatomical_data = json.load(f)
    slice_id_full = os.path.basename(os.path.dirname(reference_image_path))
    pattern = re.compile(r"^(ACDC)_patient(\d+)|^(DSB)_(\d+)")
    match = pattern.match(slice_id_full)
    if not match:
        print("[Warning] Could not parse patient info from path. Using a default prompt.")
        return "A 2D cardiac MRI at the Mid-ventricular level."
    acdc_name, acdc_id, dsb_name, dsb_id = match.groups()
    dataset = acdc_name or dsb_name
    patient_id = int(acdc_id or dsb_id)
    unique_patient_key = (dataset, patient_id)
    patient_meta = all_metadata.get(unique_patient_key, {})
    anat_level_info = anatomical_data.get(slice_id_full, {})
    prompt_parts = [
        f"A 2D cardiac MRI at the {anat_level_info.get('level_derived', 'Unknown level')} level.",
        f"The patient's diagnosis is {DIAGNOSIS_GROUP_MAPPING.get(patient_meta.get('pathology', 'Unknown'))}.",
    ]
    if pd.notna(patient_meta.get('Sex')): prompt_parts.append(f"Patient sex is {'male' if patient_meta['Sex'] == 'M' else 'female'}.")
    if pd.notna(patient_meta.get('Age')): prompt_parts.append(f"Patient age is {int(patient_meta['Age'])} years.")
    if pd.notna(patient_meta.get('height')): prompt_parts.append(f"Patient height is {patient_meta['height']:.1f} m.")
    if pd.notna(patient_meta.get('weight')): prompt_parts.append(f"Patient weight is {patient_meta['weight']:.1f} kg.")
    prompt = " ".join(prompt_parts)
    print(f"Successfully constructed prompt: '{prompt}'")
    return prompt

def load_and_preprocess_gt_sequence(gt_dir, transforms):
    """
    Loads all PNG frames from a directory, sorts them numerically,
    and applies the same preprocessing transforms used for inference.
    """
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')),
                      key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
    
    if not gt_paths:
        raise FileNotFoundError(f"No ground truth PNG files found in {gt_dir}")

    gt_frames_display = []
    for path in gt_paths:
        img_dict = transforms({"image": path})
        img_tensor = img_dict["image"]
        
        display_img = (img_tensor / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        gt_frames_display.append((display_img * 255).astype(np.uint8))
        
    return gt_frames_display

def generate_motion_sequence(config):
    os.makedirs(config.output_dir, exist_ok=True)
    
    # --- 1. Load All Necessary Models ---
    print("Loading pre-trained models...")
    vae = AutoencoderKL.from_pretrained(config.finetuned_vae_path).to(config.device)
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder", local_files_only=True).to(config.device)
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler", local_files_only=True)

    unet_flow = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet", local_files_only=True)
    new_in_channels = config.transformation_channels + config.vae_latent_channels
    unet_flow.conv_in = nn.Conv2d(new_in_channels, unet_flow.conv_in.out_channels, kernel_size=3, padding=1)
    unet_flow.conv_out = nn.Conv2d(unet_flow.conv_out.in_channels, config.transformation_channels, kernel_size=3, padding=1)
    
    print(f"Loading fine-tuned flow UNet weights from: {config.flow_unet_path}")
    weights_path = os.path.join(config.flow_unet_path, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(weights_path): raise FileNotFoundError(f"Could not find safetensors file at: {weights_path}")
    state_dict = load_file(weights_path, device="cpu")
    unet_flow.load_state_dict(state_dict)
    unet_flow = unet_flow.to(config.device)
    
    time_embedder = SinusoidalPositionalEmbedding(unet_flow.config.cross_attention_dim).to(config.device)
    vae.eval(); text_encoder.eval(); unet_flow.eval(); time_embedder.eval()


    # --- 2. Prepare Reference Image, Prompt, and Ground Truth ---
    inference_transforms = mt.Compose([
        mt.LoadImaged(keys=["image"]), mt.EnsureChannelFirstd(keys=["image"]), mt.RepeatChanneld(keys=["image"], repeats=3),
        mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=config.img_size, mode="constant", constant_values=0),
        mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
    ])

    gt_dir = os.path.dirname(config.reference_image_path)
    gt_frames_for_display = load_and_preprocess_gt_sequence(gt_dir, inference_transforms)
    num_frames_to_generate = len(gt_frames_for_display)
    print(f"Ground truth sequence found. Generating {num_frames_to_generate} frames to match.")

    print(f"Loading and preprocessing reference image: {config.reference_image_path}")
    prompt = build_inference_prompt(config.reference_image_path, config)
    ref_dict = inference_transforms({"image": config.reference_image_path})
    ref_image_tensor = ref_dict["image"].unsqueeze(0).to(config.device)

    # --- 3. The Core Inference Loop ---
    generated_frames = []
    ref_img_pil = (ref_image_tensor.squeeze(0) / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    generated_frames.append((ref_img_pil * 255).astype(np.uint8))
    
    with torch.no_grad():
        ref_latent = vae.encode(ref_image_tensor).latent_dist.sample()
        input_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(config.device)
        text_embeddings = text_encoder(input_ids)[0]
    
    time_steps = torch.linspace(0, 200, num_frames_to_generate)
    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    for i in tqdm(range(1, len(time_steps)), desc="Generating frames"):
        time_val = time_steps[i]
        latent_transformation = torch.randn(
            (1, config.transformation_channels, ref_latent.shape[2], ref_latent.shape[3]),
            generator=generator, device=config.device
        )
        noise_scheduler.set_timesteps(config.num_inference_steps)
        
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                cardiac_time_embedding = time_embedder(torch.tensor([time_val], device=config.device)).unsqueeze(1)
                cond_emb = torch.cat([text_embeddings, cardiac_time_embedding], dim=1)
                unet_input = torch.cat([latent_transformation, ref_latent], dim=1)
                noise_pred = unet_flow(unet_input, t, encoder_hidden_states=cond_emb).sample
                latent_transformation = noise_scheduler.step(noise_pred, t, latent_transformation).prev_sample

        with torch.no_grad():
            generated_flow = latent_transformation[:, :2, :, :]
            generated_residual = latent_transformation[:, 2:, :, :]
            pred_latent_norm = warp_latent(ref_latent, generated_flow) + generated_residual
            # refined_latent_norm = latent_refiner(pred_latent_norm)
            image = vae.decode(pred_latent_norm).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_frames.append((img_np * 255).astype(np.uint8))

    # --- 4. Save Outputs (Original and Comparison) ---
    print("Saving original generated outputs...")
    gif_path = os.path.join(config.output_dir, "cardiac_motion_generated.gif")
    imageio.mimsave(gif_path, generated_frames, fps=10, loop=0)
    print(f"Saved generated animated GIF to: {gif_path}")

    # Save 4D NIfTI file of the generated sequence
    grayscale_frames = [frame[:, :, 0] for frame in generated_frames]
    nifti_array = np.expand_dims(np.stack(grayscale_frames, axis=0), axis=1)
    sitk_image = sitk.GetImageFromArray(nifti_array, isVector=False)
    sitk_image.SetSpacing([1.0, 1.0, 1.0, 1000.0 / 10]) # Spacing (X,Y,Z,T), T in ms
    nifti_path = os.path.join(config.output_dir, "cardiac_motion_generated_4d.nii.gz")
    sitk.WriteImage(sitk_image, nifti_path)
    print(f"Saved 4D NIfTI file to: {nifti_path}")

    ### --- Create and save comparison video --- ###
    print("Creating and saving comparison video (Generated | Ground Truth | Error Map)...")
    comparison_frames = []
    for gen_frame, gt_frame in zip(generated_frames, gt_frames_for_display):
        # Ensure frames are single-channel for difference calculation
        gen_gray = gen_frame[:, :, 0]
        gt_gray = gt_frame[:, :, 0]

        # Calculate absolute difference and normalize for visualization
        diff = np.abs(gen_gray.astype(np.float32) - gt_gray.astype(np.float32))
        if diff.max() > 0:
            diff_normalized = (diff - diff.min()) / (diff.max() - diff.min())
        else:
            diff_normalized = np.zeros_like(diff, dtype=np.float32)

        # Apply a colormap to the error map and convert to uint8 RGB
        error_map_rgb = (plt.cm.viridis(diff_normalized)[:, :, :3] * 255).astype(np.uint8)

        # Concatenate horizontally: [Generated | GT | Error]
        combined_frame = np.hstack((gen_frame, gt_frame, error_map_rgb))
        comparison_frames.append(combined_frame)

    # Save the new comparison GIF
    comparison_gif_path = os.path.join(config.output_dir, "comparison_motion.gif")
    imageio.mimsave(comparison_gif_path, comparison_frames, fps=10, loop=0)
    print(f"Saved comparison GIF to: {comparison_gif_path}")


if __name__ == "__main__":
    config = InferenceConfig()
    generate_motion_sequence(config)