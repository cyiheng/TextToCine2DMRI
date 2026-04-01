import math
import os
import sys
import shutil
import glob
import re
import json
import random
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from monai import transforms as mt
from monai.config import print_config
from monai.utils import set_determinism
import matplotlib.pyplot as plt

print_config()
set_determinism(42)

# --- Configuration ---
class Config:
    # Paths
    base_data_dir = "./data/" 
    dataset_root = "./data/dataset/"
    anatomical_json_path = "./data/anatomical_levels.json"
    finetuned_vae_path = "./results/001_vae_finetuned/"
    output_dir = "./results/003_FirstFrameSD/"
    # Set to "latest" to automatically resume from the latest checkpoint in output_dir
    # Or provide a specific path like "./103_StableDiffusionFT/002_unet_finetuned/unet_epoch_150"
    resume_from_checkpoint = "latest" 

    # Model
    model_name = "runwayml/stable-diffusion-v1-5"
    
    # Data
    img_size = (192, 192)
    # Use the first 100 patients from ACDC for training
    acdc_train_patients = 100
    # Use the next 20 patients from ACDC for validation
    acdc_val_patients = 10
    
    # Use the first 500 patients from DSB for training
    dsb_train_patients = 500
    # Use the next 50 patients from DSB for validation
    dsb_val_patients = 10
    
    # Training
    max_train_steps = 200000          # The total number of training steps to perform.
    batch_size = 8 
    learning_rate = 1e-4
    lr_scheduler_type = "constant_with_warmup"
    lr_warmup_steps = 200
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    
    # Classifier-Free Guidance (CFG)
    cfg_dropout_prob = 0.2 

    # Validation
    validation_steps = 100000           # Run validation every 5000 steps.
    checkpointing_steps = 100000        # Save a checkpoint every 5000 steps.

    # Plotting
    plot_smoothing_window = 100

config = Config()
os.makedirs(config.output_dir, exist_ok=True)
# --- Utility Mappings ---
DIAGNOSIS_GROUP_MAPPING = {
    "NOR": "Normal function", "MINF": "Myocardial Infarction", "DCM": "Dilated Cardiomyopathy",
    "HCM": "Hypertrophic Cardiomyopathy", "RV": "Abnormal Right Ventricle",
}

# --- Generic Metadata Loader ---
def load_all_patient_metadata(base_data_dir):
    """
    Loads and consolidates metadata from multiple datasets (ACDC, DSB) by
    iterating through their train/test/val metadata CSVs.
    """
    print("--- Loading and consolidating all patient metadata ---")
    all_metadata = {}

    datasets_config = {
        'ACDC': {
            'folder': os.path.join(base_data_dir, 'ACDC_Preprocessed'),
            'id_col': 'pid',
            'info_cols': ["pathology","height","weight"]
        },
        'DSB': {
            'folder': os.path.join(base_data_dir, 'DSB_nifti'),
            'id_col': 'pid',
            'info_cols': ['Sex', 'Age']
        }
    }

    for dataset_name, params in datasets_config.items():
        print(f"-> Processing {dataset_name} metadata...")
        for split in ['train', 'test', 'val']:
            csv_path = os.path.join(params['folder'], f"{split}_metadata.csv")
            if dataset_name == "DSB":
                csv_path = os.path.join(params['folder'], f"{split}_metadata_additional_with_es.csv")
                
            if not os.path.exists(csv_path):
                continue
            
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    patient_id_val = row[params['id_col']]
                    
                    # Extract numeric patient ID
                    match = re.search(r"(\d+)", str(patient_id_val))
                    if not match: continue
                    patient_num = int(match.group(1))
                    
                    # Create the unique key and store the relevant info
                    unique_patient_key = (dataset_name, patient_num)
                    patient_info = {col: row.get(col) for col in params['info_cols']}
                    all_metadata[unique_patient_key] = patient_info

            except Exception as e:
                print(f"  [Error] Failed to process {csv_path}: {e}")

    print(f"Loaded metadata for {len(all_metadata)} unique patients from all sources.")
    return all_metadata


# --- Data Preparation Function ---
def prepare_datalist_first_frame(cfg):
    print("--- Preparing Datalist for First Frame Fine-tuning (ACDC & DSB) ---")
    
    # Pre-load all metadata using the new generic function
    all_patient_metadata = load_all_patient_metadata(cfg.base_data_dir)
    with open(cfg.anatomical_json_path, 'r') as f:
        anatomical_data = json.load(f)
    
    datalist = []
    pattern = re.compile(r"^(ACDC)_patient(\d+)|^(DSB)_(\d+)")
    all_slice_dirs = glob.glob(os.path.join(cfg.dataset_root, "*"))

    for slice_dir_path in tqdm(all_slice_dirs, desc="Scanning slices"):
        slice_id_full = os.path.basename(slice_dir_path)
        match = pattern.match(slice_id_full)
        if not match: continue

        acdc_name, acdc_id, dsb_name, dsb_id = match.groups()
        dataset = acdc_name or dsb_name
        patient_id = int(acdc_id or dsb_id)
        unique_patient_key = (dataset, patient_id)

        if unique_patient_key not in all_patient_metadata: continue

        first_frame_path = os.path.join(slice_dir_path, "frame_000.png")
        if not os.path.exists(first_frame_path): continue

        patient_info = all_patient_metadata[unique_patient_key]
        anat_level_info = anatomical_data.get(slice_id_full)
        if not anat_level_info: continue
            
        datalist.append({
            "image": first_frame_path, "unique_patient_key": unique_patient_key,
            "anatomical_level": anat_level_info.get("level_derived", "Unknown level"),
            "diagnosis_group": DIAGNOSIS_GROUP_MAPPING.get(patient_info.get('pathology', 'Unknown')),
            "height": patient_info.get('height'), "weight": patient_info.get('weight'),
            "sex": patient_info.get('Sex'), "age": patient_info.get('Age'),
        })

    # --- Patient-Aware Splitting ---
    patient_data = defaultdict(list)
    for item in datalist:
        patient_data[item['unique_patient_key']].append(item)

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
    print(f"Total first frames found and processed: {len(datalist)}")
    print(f"Training Patients: {len(train_pids)}, Training Samples: {len(train_data)}")
    print(f"Validation Patients: {len(val_pids)}, Validation Samples: {len(val_data)}")
    print("-" * 55)
    return train_data, val_data

# --- Dataset Class ---
class FirstFrameDataset(torch.utils.data.Dataset):
    def __init__(self, datalist, tokenizer, transform):
        self.datalist = datalist
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        transformed = self.transform({"image": item['image']})
        pixel_values = transformed["image"]

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
        return {"pixel_values": pixel_values, "input_ids": input_ids.squeeze()}

# --- Main Training Function ---
def main():
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
    
    # --- Model Loading ---
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(config.finetuned_vae_path)
    unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet")
    vae.requires_grad_(False); text_encoder.requires_grad_(False)
    
    # --- Data Preparation ---
    train_data, val_data = prepare_datalist_first_frame(config)
    
    train_transforms = mt.Compose([
        mt.LoadImaged(keys=["image"]), mt.EnsureChannelFirstd(keys=["image"]),
        mt.RepeatChanneld(keys=["image"], repeats=3),
        mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=config.img_size, mode="constant", constant_values=0),
        # mt.RandAffined(keys=["image"], prob=0.5, rotate_range=(0.1), padding_mode="zeros"),
        mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
    ])

    train_dataset = FirstFrameDataset(train_data, tokenizer, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    # --- Optimizer ---
    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
        print("Using 8-bit AdamW optimizer.")
    except ImportError:
        optimizer_cls = torch.optim.AdamW
        print("bitsandbytes not found. Using standard AdamW optimizer.")

    optimizer = optimizer_cls(
        unet.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device); text_encoder.to(accelerator.device)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    # Re-calculate the number of epochs required for the total steps
    max_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    
    # --- Resuming Logic ---
    starting_epoch = 0
    resume_step = 0
    if config.resume_from_checkpoint:
        path = None
        if config.resume_from_checkpoint != "latest":
            path = config.resume_from_checkpoint
        else:
            dirs = os.listdir(config.output_dir)
            checkpoint_dirs = [d for d in dirs if d.startswith("checkpoint_step_")]
            if len(checkpoint_dirs) > 0:
                # Sort by the step number in the directory name
                checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
                path = os.path.join(config.output_dir, checkpoint_dirs[-1])
                accelerator.print(f"Found latest resumable checkpoint: {path}")

        if path and os.path.exists(path):
            try:
                accelerator.print(f"Attempting to resume full state from {path}...")
                accelerator.load_state(path)
                
                # --- Determine resume step from path ---
                step_match = re.search(r"step_(\d+)", os.path.basename(path))
                if step_match:
                    resume_step = int(step_match.group(1))
                    starting_epoch = resume_step // num_update_steps_per_epoch
                    accelerator.print(f"Successfully resumed state. Will continue from step {resume_step} (approx. epoch {starting_epoch}).")
                else:
                     accelerator.print("Warning: Could not determine step from checkpoint path. Resuming from step 0.")

            except (FileNotFoundError, OSError, ValueError) as e:
                accelerator.print(f"Could not load full state from {path} (error: {e}). Starting from scratch.")
        else:
            accelerator.print("No valid checkpoint found to resume from. Starting from scratch.")

    if accelerator.is_main_process:
        step_loss_list = []
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {max_train_epochs} (calculated)")
    print(f"  Total optimization steps = {config.max_train_steps}")

    progress_bar = tqdm(
        range(config.max_train_steps), 
        initial=resume_step, # Start the progress bar from the resumed step
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    global_step = resume_step
    
    # The outer epoch loop is kept for DataLoader compatibility, but the break condition is step-based
    for epoch in range(starting_epoch, max_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # The actual training logic for a single step
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                if random.random() < config.cfg_dropout_prob:
                    uncond_tokens = tokenizer([""] * bsz, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
                    uncond_embeddings = text_encoder(uncond_tokens.input_ids.to(accelerator.device))[0]
                    encoder_hidden_states = uncond_embeddings

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --- Step-based progress tracking and checkpointing ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                avg_loss = accelerator.gather(loss.detach()).mean()
                if accelerator.is_main_process:
                    step_loss_list.append(avg_loss.item())
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            # --- Step-based validation and checkpointing ---
            if accelerator.is_main_process:
                # Check for validation
                if global_step > 0 and global_step % config.validation_steps == 0:
                    print(f"\nStep {global_step}: Running validation...")
                    
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        config.model_name,
                        vae=accelerator.unwrap_model(vae),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(unet),
                        safety_checker=None,
                        torch_dtype=torch.float16,
                    ).to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    
                    val_prompts = [
                        "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Normal function.",
                        "A 2D cardiac MRI at the Apical level, shows End-Diastole for a patient with Hypertrophic Cardiomyopathy.",
                        "A 2D cardiac MRI at the Basal level, shows End-Diastole for a patient with Myocardial Infarction.",
                        "A 2D cardiac MRI at the Mid-ventricular level, shows End-Diastole for a patient with Dilated Cardiomyopathy.",
                    ]
                    generator = torch.Generator(device=accelerator.device).manual_seed(42)
                    
                    with torch.no_grad():
                        images = pipeline(val_prompts, num_inference_steps=30, generator=generator, height=config.img_size[0], width=config.img_size[1]).images
                    
                    fig, axs = plt.subplots(1, len(images), figsize=(len(images)*4, 4))
                    for i, img in enumerate(images): axs[i].imshow(img, cmap="gray"); axs[i].axis("off")
                    plt.tight_layout()
                    val_dir = os.path.join(config.output_dir, "validation_samples")
                    os.makedirs(val_dir, exist_ok=True)
                    plt.savefig(f"{val_dir}/step_{global_step:06d}.png")
                    plt.close()
                    del pipeline # Free up memory
                    torch.cuda.empty_cache()

                # Check for checkpointing
                if global_step > 0 and global_step % config.checkpointing_steps == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint_step_{global_step}")
                    accelerator.save_state(save_path)
                    print(f"Saved resumable state to {save_path}")

                    # Optionally save the UNet separately for easier inference
                    unet_save_path = os.path.join(config.output_dir, f"unet_step_{global_step}")
                    accelerator.unwrap_model(unet).save_pretrained(unet_save_path)
                    print(f"Saved UNet for inference to {unet_save_path}")
            
            # --- Break condition to stop training ---
            if global_step >= config.max_train_steps:
                break
        
        # Also break the outer loop
        if global_step >= config.max_train_steps:
            break

    # --- Final save ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Final learning curve plot
        plt.figure("Learning Curve", figsize=(12, 6))
        plt.plot(step_loss_list, label="Step Loss", alpha=0.5)
        if len(step_loss_list) > config.plot_smoothing_window:
            smoothed_loss = np.convolve(step_loss_list, np.ones(config.plot_smoothing_window)/config.plot_smoothing_window, mode='valid')
            plt.plot(np.arange(config.plot_smoothing_window - 1, len(step_loss_list)), smoothed_loss, label=f"Smoothed Loss (window={config.plot_smoothing_window})", color='red')
        plt.title("UNet Fine-tuning Loss Curve"); plt.xlabel("Training Step"); plt.ylabel("MSE Loss")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(config.output_dir, "learning_curve_final.png"))
        plt.close()

        # Save the final model
        final_save_path = os.path.join(config.output_dir, f"checkpoint_step_{global_step}_final")
        accelerator.save_state(final_save_path)
        unet_final = accelerator.unwrap_model(unet)
        unet_final.save_pretrained(os.path.join(config.output_dir, "unet_final"))
        print(f"Final fine-tuned state and UNet saved after {global_step} steps.")

if __name__ == "__main__":
    
    try:
        source_script_path = sys.argv[0]
        script_name = os.path.basename(source_script_path)
        dest_script_path = os.path.join(config.output_dir, script_name)
        shutil.copy(source_script_path, dest_script_path)
        print(f"Copied training script '{source_script_path}' to '{dest_script_path}'")
    except Exception as e:
        print(f"Warning: Could not copy training script. Error: {e}")
    main()
    # accelerate launch train_stage3.py