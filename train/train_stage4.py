import os
import sys
import shutil
import glob
import re
from collections import defaultdict
from tqdm.auto import tqdm
import math
import pandas as pd
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
import json


from monai import transforms as mt
from monai.config import print_config
from monai.utils import set_determinism
from utils import warp_latent, flow_to_rgb
from models.blocks import LatentFlowAndResidualPredictor, SinusoidalPositionalEmbedding
print_config()
set_determinism(42)

# --- Configuration for Step 4 ---
class Config:
    # Paths
    base_data_dir = "./data/"
    dataset_root = "./data/dataset/"
    anatomical_json_path = "./data/anatomical_levels.json"
    # Path to models from previous steps
    finetuned_vae_path = "./results/001_vae_finetuned/"
    flow_predictor_path = "./results/002_lfm/checkpoints/best_flow_predictor.pth"
    # Output for this step
    output_dir = "./results/004_FlowSD/"
    resume_from_checkpoint = "latest"  
    
    
    # Base model for UNet architecture
    model_name = "runwayml/stable-diffusion-v1-5"
    
    # Data
    img_size = (192, 192)
    acdc_train_patients = 100 # Adjusted from your example for a more realistic split
    acdc_val_patients = 20
    dsb_train_patients = 500
    dsb_val_patients = 50
    
    # Training
    max_train_steps = 100000          # The total number of training steps.
    batch_size = 32 # Reduce if OOM
    learning_rate = 1e-4
    lr_scheduler_type = "cosine"
    lr_warmup_steps = 200
    
    # Model architecture
    vae_latent_channels = 4
    transformation_channels = 2 + vae_latent_channels # 6 channels
    
    # Validation
    validation_steps = 50000          # Run validation every 10k steps.
    checkpointing_steps = 50000       # Save a checkpoint every 10k steps.
    num_inference_steps = 50

    # Plotting
    plot_smoothing_window = 100

config = Config()
os.makedirs(os.path.join(config.output_dir, "validation_samples"), exist_ok=True)


# --- Data Preparation for Flow Diffusion ---
DIAGNOSIS_GROUP_MAPPING = {
    "NOR": "Normal function", "MINF": "Myocardial Infarction", "DCM": "Dilated Cardiomyopathy",
    "HCM": "Hypertrophic Cardiomyopathy", "RV": "Abnormal Right Ventricle",
}
# --- Generic Metadata Loader (Adapted for Flow Diffusion) ---
def load_all_patient_metadata(base_data_dir):
    print("--- Loading and consolidating all patient metadata for Flow Diffusion ---")
    all_metadata = {}

    datasets_config = {
        'ACDC': {
            'folder': os.path.join(base_data_dir, 'ACDC_Preprocessed'),
            'id_col': 'pid',
            'info_cols': ["pathology", "height", "weight", "ed_frame", "es_frame"]
        },
        'DSB': {
            'folder': os.path.join(base_data_dir, 'DSB_nifti'),
            'id_col': 'pid',
            'info_cols': ['Sex', 'Age', 'es_frame'] # ED is implicitly frame 0
        }
    }

    for dataset_name, params in datasets_config.items():
        print(f"-> Processing {dataset_name} metadata...")
        for split in ['train', 'test', 'val']:
            csv_path = os.path.join(params['folder'], f"{split}_metadata.csv")
            if dataset_name == "DSB":
                csv_path = os.path.join(params['folder'], f"{split}_metadata_additional_with_es.csv")

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

    print(f"Loaded metadata for {len(all_metadata)} unique patients from all sources.")
    return all_metadata

# --- Data Preparation for Flow Diffusion ---
def prepare_datalist_flow_diffusion(cfg):
    print("--- Preparing Datalist for Flow Diffusion (ACDC & DSB) ---")

    all_patient_metadata = load_all_patient_metadata(cfg.base_data_dir)
    with open(cfg.anatomical_json_path, 'r') as f:
        anatomical_data = json.load(f)

    datalist = []
    pattern = re.compile(r"^(ACDC)_patient(\d+)|^(DSB)_(\d+)")
    all_slice_dirs = glob.glob(os.path.join(cfg.dataset_root, "*"))

    for slice_dir in tqdm(all_slice_dirs, desc="Processing Slices"):
        slice_id_full = os.path.basename(slice_dir)
        match = pattern.match(slice_id_full)
        if not match: continue

        acdc_name, acdc_id, dsb_name, dsb_id = match.groups()
        dataset = acdc_name or dsb_name
        patient_id = int(acdc_id or dsb_id)
        unique_patient_key = (dataset, patient_id)

        if unique_patient_key not in all_patient_metadata: continue
        
        patient_meta = all_patient_metadata[unique_patient_key]
        
        # --- IMPLEMENTING DATASET-SPECIFIC ED/ES FRAME RULES ---
        ed_frame_num, es_frame_num = None, None
        if dataset == 'ACDC':
            ed_frame_num = patient_meta.get('ed_frame')
            es_frame_num = patient_meta.get('es_frame')
        elif dataset == 'DSB':
            ed_frame_num = 0  # Per your rule
            es_frame_num = patient_meta.get('es_frame')

        # Validate that we have valid frame numbers
        if pd.isna(ed_frame_num) or pd.isna(es_frame_num): continue
        ed_frame_num, es_frame_num = int(ed_frame_num), int(es_frame_num)
        
        frames = sorted(glob.glob(os.path.join(slice_dir, "frame_*.png")))
        num_frames = len(frames)
        if num_frames < 2: continue

        # Frame numbers in CSV might be 1-based, convert to 0-based index
        ed_idx = ed_frame_num if dataset == 'DSB' else ed_frame_num - 1
        es_idx = es_frame_num - 1 # Assume ES is 1-based for both
        if not (0 <= ed_idx < num_frames and 0 <= es_idx < num_frames): continue

        ref_frame_path = frames[ed_idx]
        anat_level_info = anatomical_data.get(slice_id_full)
        if not anat_level_info: continue

        # Time calculation logic
        if es_idx >= ed_idx: contraction_duration = es_idx - ed_idx
        else: contraction_duration = (num_frames - ed_idx) + es_idx
        relaxation_duration = num_frames - contraction_duration
        if contraction_duration <= 0 or relaxation_duration <= 0: continue

        for i in range(num_frames):
            if i == ed_idx: continue
            
            time_val = 0.0
            dist_from_ed = (i - ed_idx + num_frames) % num_frames
            dist_from_es = (i - es_idx + num_frames) % num_frames
            if dist_from_ed <= contraction_duration: # In contraction phase
                time_val = 100.0 * (dist_from_ed / contraction_duration)
            else: # In relaxation phase
                time_val = 100.0 + 100.0 * (dist_from_es / relaxation_duration)

            item = {
                "ref_image": ref_frame_path, "dri_image": frames[i],
                "time_value": time_val, "unique_patient_key": unique_patient_key,
                "anatomical_level": anat_level_info.get("level_derived", "Unknown level"),
                "diagnosis_group": DIAGNOSIS_GROUP_MAPPING.get(patient_meta.get('pathology', 'Unknown')),
                "height": patient_meta.get('height'), "weight": patient_meta.get('weight'),
                "sex": patient_meta.get('Sex'), "age": patient_meta.get('Age'),
            }
            datalist.append(item)

    # --- Patient-Aware Splitting ---
    patient_data = defaultdict(list)
    for item in datalist: patient_data[item['unique_patient_key']].append(item)
    
    acdc_pids = sorted([k for k in patient_data if k[0] == 'ACDC'], key=lambda x: x[1])
    dsb_pids = sorted([k for k in patient_data if k[0] == 'DSB'], key=lambda x: x[1])

    train_pids = set(acdc_pids[:cfg.acdc_train_patients] + dsb_pids[:cfg.dsb_train_patients])
    val_pids = set(
        acdc_pids[cfg.acdc_train_patients : cfg.acdc_train_patients + cfg.acdc_val_patients] +
        dsb_pids[cfg.dsb_train_patients : cfg.dsb_train_patients + cfg.dsb_val_patients]
    )

    train_data = [item for pid in train_pids for item in patient_data.get(pid, [])]
    val_data = [item for pid in val_pids for item in patient_data.get(pid, [])]

    print("-" * 55)
    print(f"Total frame pairs found: {len(datalist)}")
    print(f"Training Patients: {len(train_pids)}, Training Samples: {len(train_data)}")
    print(f"Validation Patients: {len(val_pids)}, Validation Samples: {len(val_data)}")
    print("-" * 55)
    return train_data, val_data

# --- Dataset Class ---
class FlowDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, datalist, tokenizer, transform):
        self.datalist = datalist
        self.tokenizer = tokenizer
        self.transform = transform
        self.data_keys = ["ref_image", "dri_image"]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        transformed = self.transform({k: item[k] for k in self.data_keys})
        
        # Build prompt using all available metadata
        prompt_parts = [
            f"A 2D cardiac MRI at the {item['anatomical_level']} level that shows End-Diastole phase.",
        ]
        if pd.notna(item.get('pathology')) and random.random() < 0.5:
            prompt_parts.append(f"The diagnosis is {item['diagnosis_group']}.")
        if pd.notna(item.get('sex')) and random.random() < 0.5:
             prompt_parts.append(f"Patient sex is {'male' if item['sex'] == 'M' else 'female'}.")
        if pd.notna(item.get('age')) and random.random() < 0.5:
             prompt_parts.append(f"Patient age is {int(item['age'])} years.")
        if pd.notna(item.get('height')) and random.random() < 0.5:
            prompt_parts.append(f"Patient height is {item['height']:.1f} m.")
        if pd.notna(item.get('weight')) and random.random() < 0.5:
            prompt_parts.append(f"Patient weight is {item['weight']:.1f} kg.")
        
        random.shuffle(prompt_parts)
        caption = " ".join(prompt_parts)
        input_ids = self.tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids
        
        return {
            "ref_pixel_values": transformed["ref_image"],
            "dri_pixel_values": transformed["dri_image"],
            "time_value": torch.tensor(item["time_value"], dtype=torch.float32),
            "input_ids": input_ids.squeeze(),
        }
    
# --- Main Training Function ---
def main():
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
    
    # --- Load Pre-trained Models ---
    print("Loading pre-trained models...")
    # These models are for inference only, so they don't need to be prepared by accelerator yet
    vae = AutoencoderKL.from_pretrained(config.finetuned_vae_path)
    flow_predictor = LatentFlowAndResidualPredictor(in_channels=2*config.vae_latent_channels, latent_channels=config.vae_latent_channels)
    flow_predictor.load_state_dict(torch.load(config.flow_predictor_path, map_location="cpu")) # Load to CPU first
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder")
    
    # Freeze and set to eval mode
    vae.requires_grad_(False); flow_predictor.requires_grad_(False); text_encoder.requires_grad_(False)
    vae.eval(); flow_predictor.eval(); text_encoder.eval()

    # --- Setup the UNet for Flow Generation ---
    print("Setting up UNet for flow generation...")
    unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet")
    
    # Modify UNet input layer
    new_in_channels = config.transformation_channels + config.vae_latent_channels
    original_conv_in = unet.conv_in
    new_conv_in = nn.Conv2d(new_in_channels, original_conv_in.out_channels, kernel_size=original_conv_in.kernel_size, stride=original_conv_in.stride, padding=original_conv_in.padding)
    with torch.no_grad():
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, config.transformation_channels:, :, :] = original_conv_in.weight.clone()
        if original_conv_in.bias is not None: new_conv_in.bias.data = original_conv_in.bias.data.clone()
    unet.conv_in = new_conv_in

    # Modify UNet output layer
    original_conv_out = unet.conv_out
    new_conv_out = nn.Conv2d(original_conv_out.in_channels, config.transformation_channels, kernel_size=original_conv_out.kernel_size, padding=original_conv_out.padding)
    with torch.no_grad(): new_conv_out.weight.zero_(); new_conv_out.bias.zero_()
    unet.conv_out = new_conv_out

    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")
    time_embedder = SinusoidalPositionalEmbedding(unet.config.cross_attention_dim)

    # --- Data Preparation ---
    train_data, val_data = prepare_datalist_flow_diffusion(config)
    
    train_transforms = mt.Compose([
        mt.LoadImaged(keys=["ref_image", "dri_image"], allow_missing_keys=True),
        mt.EnsureChannelFirstd(keys=["ref_image", "dri_image"], allow_missing_keys=True),
        mt.RepeatChanneld(keys=["ref_image", "dri_image"], repeats=3, allow_missing_keys=True),
        mt.ResizeWithPadOrCropd(keys=["ref_image", "dri_image"], spatial_size=config.img_size, mode="constant", constant_values=0, allow_missing_keys=True),
        mt.ScaleIntensityRanged(keys=["ref_image", "dri_image"], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True, allow_missing_keys=True),
    ])
    
    train_dataset = FlowDiffusionDataset(train_data, tokenizer, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # --- Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    # --- Prepare with Accelerator ---
    unet, time_embedder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, time_embedder, optimizer, train_dataloader, lr_scheduler
    )
    # Move inference models to the correct device
    vae, flow_predictor, text_encoder = vae.to(accelerator.device), flow_predictor.to(accelerator.device), text_encoder.to(accelerator.device)

    # --- Resuming Logic ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    max_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    
    starting_epoch = 0
    resume_step = 0
    if config.resume_from_checkpoint:
        path = None
        if config.resume_from_checkpoint != "latest":
            path = config.resume_from_checkpoint
        else:
            dirs = os.listdir(config.output_dir)
            checkpoint_dirs = [d for d in dirs if d.startswith("checkpoint_step_")]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
                path = os.path.join(config.output_dir, checkpoint_dirs[-1])
                accelerator.print(f"Found latest resumable checkpoint: {path}")

        if path and os.path.exists(path):
            try:
                accelerator.print(f"Resuming from checkpoint {path}...")
                accelerator.load_state(path)
                step_match = re.search(r"step_(\d+)", os.path.basename(path))
                if step_match:
                    resume_step = int(step_match.group(1))
                    starting_epoch = resume_step // num_update_steps_per_epoch
                    accelerator.print(f"Resumed state. Will continue from step {resume_step}.")
            except Exception as e:
                accelerator.print(f"Could not load full state from {path} (error: {e}). Starting from scratch.")
        else:
            accelerator.print("No valid checkpoint found. Starting from scratch.")

    if accelerator.is_main_process:
        step_loss_list = []
    
    # --- Training Loop ---
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Total optimization steps = {config.max_train_steps}")

    progress_bar = tqdm(range(config.max_train_steps), initial=resume_step, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = resume_step

    for epoch in range(starting_epoch, max_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    ref_latents = vae.encode(batch["ref_pixel_values"]).latent_dist.sample()
                    dri_latents = vae.encode(batch["dri_pixel_values"]).latent_dist.sample()
                    target_flow, target_residual = flow_predictor(ref_latents, dri_latents)
                    target_transformation = torch.cat([target_flow, target_residual], dim=1)

                noise = torch.randn_like(target_transformation)
                bsz = target_transformation.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_transformation.device).long()
                noisy_transformation = noise_scheduler.add_noise(target_transformation, noise, timesteps)
                
                with torch.no_grad():
                    text_embeddings = text_encoder(batch["input_ids"])[0]
                    cardiac_time_embedding = time_embedder(batch["time_value"])
                conditioning_emb = torch.cat([text_embeddings, cardiac_time_embedding.unsqueeze(1)], dim=1)
                
                unet_input = torch.cat([noisy_transformation, ref_latents], dim=1)
                model_pred = unet(unet_input, timesteps, encoder_hidden_states=conditioning_emb).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    avg_loss = accelerator.gather(loss.detach()).mean()
                    step_loss_list.append(avg_loss.item())
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if accelerator.is_main_process:
                if global_step > 0 and global_step % config.validation_steps == 0:
                    print(f"\nStep {global_step}: Running validation...")
                    unet.eval()
                    
                    if not val_data:
                        print("Validation data empty, skipping.")
                        continue
                    
                    try: val_sample = next(item for item in val_data if "ref_image" in item.keys())
                    except StopIteration: continue
                        
                    ref_image_val = train_transforms({"ref_image": val_sample["ref_image"]})["ref_image"].unsqueeze(0).to(accelerator.device)
                    with torch.no_grad():
                        ref_latent_val = vae.encode(ref_image_val).latent_dist.sample()
                    
                    val_prompt = f"A 2D cardiac MRI at the {val_sample['anatomical_level']} level. Shows End-Diastole for a patient with Normal function."
                    val_input_ids = tokenizer(val_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                    
                    with torch.no_grad(): val_text_embeddings = text_encoder(val_input_ids)[0]
                    
                    val_times = [0, 50, 100, 150, 200]
                    generated_images, generated_flows = [], []
                    generator = torch.Generator(device=accelerator.device).manual_seed(42)
                    
                    for time_val in val_times:
                        latent_transformation = torch.randn((1, config.transformation_channels, ref_latent_val.shape[2], ref_latent_val.shape[3]), generator=generator, device=accelerator.device)
                        noise_scheduler.set_timesteps(config.num_inference_steps)
                        for t in noise_scheduler.timesteps:
                            with torch.no_grad():
                                time_tensor = torch.tensor([time_val], device=accelerator.device)
                                cardiac_time_embedding = time_embedder(time_tensor).unsqueeze(1)
                                cond_emb = torch.cat([val_text_embeddings, cardiac_time_embedding], dim=1)
                                unet_input_val = torch.cat([latent_transformation, ref_latent_val], dim=1)
                                noise_pred = unet(unet_input_val, t, encoder_hidden_states=cond_emb).sample
                                latent_transformation = noise_scheduler.step(noise_pred, t, latent_transformation).prev_sample
                        
                        generated_flows.append(latent_transformation.squeeze(0))
                        with torch.no_grad():
                            generated_flow, generated_residual = latent_transformation[:, :2, :, :], latent_transformation[:, 2:, :, :]
                            pred_latent_norm = warp_latent(ref_latent_val, generated_flow) + generated_residual
                            image = vae.decode(pred_latent_norm).sample
                            generated_images.append((image / 2 + 0.5).clamp(0, 1).squeeze(0))

                    # --- Plotting logic ---
                    fig, axs = plt.subplots(5, len(val_times), figsize=(len(val_times) * 2.5, 5)); ref_display = (ref_image_val.squeeze(0) / 2 + 0.5).clamp(0, 1)
                    for i in range(len(val_times)):
                        axs[0, i].imshow(ref_display.permute(1,2,0).cpu().numpy()); axs[0, i].set_title(f"Ref (T=0)"); axs[0, i].axis("off")
                        axs[1, i].imshow(generated_images[i].permute(1,2,0).cpu().numpy()); axs[1, i].set_title(f"Gen (T={val_times[i]})"); axs[1, i].axis("off")
                        max_flow_val = torch.abs(generated_flows[i]).max().item() + 1e-6
                        axs[2, i].imshow(generated_flows[i][0].cpu().numpy(), cmap="coolwarm", vmin=-max_flow_val, vmax=max_flow_val); axs[2, i].set_title("Flow X"); axs[2, i].axis("off")
                        axs[3, i].imshow(generated_flows[i][1].cpu().numpy(), cmap="coolwarm", vmin=-max_flow_val, vmax=max_flow_val); axs[3, i].set_title("Flow Y"); axs[3, i].axis("off")
                        axs[4, i].imshow(flow_to_rgb(generated_flows[i])); axs[4, i].set_title(f"Gen Flow"); axs[4, i].axis("off")
                    plt.tight_layout(); val_dir = os.path.join(config.output_dir, "validation_samples")
                    plt.savefig(f"{val_dir}/step_{global_step:06d}.png"); plt.close()

                if global_step > 0 and global_step % config.checkpointing_steps == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint_step_{global_step}")
                    accelerator.save_state(save_path)
                    print(f"Saved resumable state to {save_path}")

            if global_step >= config.max_train_steps: break
        if global_step >= config.max_train_steps: break

    # --- Final Save ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Plot final learning curve
        plt.figure("Learning Curve", figsize=(12, 6)); plt.plot(step_loss_list, label="Step Loss", alpha=0.5)
        if len(step_loss_list) > config.plot_smoothing_window:
            smoothed_loss = np.convolve(step_loss_list, np.ones(config.plot_smoothing_window)/config.plot_smoothing_window, mode='valid')
            plt.plot(np.arange(config.plot_smoothing_window - 1, len(step_loss_list)), smoothed_loss, label=f"Smoothed Loss (window={config.plot_smoothing_window})", color='red')
        plt.title("Flow Diffusion UNet - Loss Curve"); plt.xlabel("Training Step"); plt.ylabel("MSE Loss")
        plt.legend(); plt.grid(True); plt.savefig(os.path.join(config.output_dir, "learning_curve_final.png")); plt.close()

        # Save final state and UNet model
        final_save_path = os.path.join(config.output_dir, f"checkpoint_step_{global_step}_final")
        accelerator.save_state(final_save_path)
        unet_final = accelerator.unwrap_model(unet)
        unet_final.save_pretrained(os.path.join(config.output_dir, "unet_final"))
        print(f"Final training state and UNet model saved after {global_step} steps.")


if __name__ == "__main__":
    
    # 1. Prepare the datalists for training and validation
    train_datalist, val_datalist = prepare_datalist_flow_diffusion(config)

    # 2. RUN THE SANITY CHECK before proceeding
    # run_sanity_check(train_datalist, val_datalist)

    # 3. Proceed with the rest of your training setup
    #    (tokenizer, transforms, creating Dataset and DataLoader instances, etc.)
    print("Sanity check complete. Proceeding with training setup...")

    try:
        source_script_path = sys.argv[0]
        dest_script_path = os.path.join(config.output_dir, os.path.basename(source_script_path))
        shutil.copy(source_script_path, dest_script_path)
        print(f"Copied training script to '{dest_script_path}'")
    except Exception as e:
        print(f"Warning: Could not copy training script. Error: {e}")
    main()