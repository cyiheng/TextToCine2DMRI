
import sys
import shutil
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL
from monai.losses import PerceptualLoss
from monai import transforms as mt
from monai.config import print_config
from monai.utils import first, set_determinism
from utils import prepare_datalists

print_config()
set_determinism(42)

# --- Configuration ---
class Config:
    dataset_root = "./data/dataset/"
    output_dir = "./results/001_vae_finetuned/"
    model_name = "runwayml/stable-diffusion-v1-5"
    img_size = (192, 192)
    
    # --- Dataset Split Configuration ---
    # Use the first 100 patients from ACDC for training
    acdc_train_patients = 100
    # Use the next 20 patients from ACDC for validation
    acdc_val_patients = 20
    
    # Use the first 500 patients from DSB for training
    dsb_train_patients = 500
    # Use the next 50 patients from DSB for validation
    dsb_val_patients = 50
    # ----------------------------------------

    epochs = 10
    batch_size = 8
    learning_rate = 1e-5
    perceptual_weight = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_interval = 5

config = Config()

# --- Setup and Data Splitting ---
os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)


class CardiacMRIDataset(Dataset):
    def __init__(self, file_list, transform):
        self.image_paths = file_list
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        data = {"image": img_path}
        return self.transform(data)

try:
    source_script_path = sys.argv[0]
    script_name = os.path.basename(source_script_path)
    dest_script_path = os.path.join(config.output_dir, script_name)
    shutil.copy(source_script_path, dest_script_path)
    print(f"Copied training script '{source_script_path}' to '{dest_script_path}'")
except Exception as e:
    print(f"Warning: Could not copy training script. Error: {e}")

train_files, val_files = prepare_datalists(
    root_dir=config.dataset_root,
    acdc_train_count=config.acdc_train_patients,
    acdc_val_count=config.acdc_val_patients,
    dsb_train_count=config.dsb_train_patients,
    dsb_val_count=config.dsb_val_patients
)

# Define normalization values once
subtrahend = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
divisor = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

train_transforms = mt.Compose([
    mt.LoadImaged(keys=["image"]),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.RepeatChanneld(keys=["image"], repeats=3),
    mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=config.img_size, mode="constant", constant_values=0),
    mt.RandAffined(
        keys=["image"], prob=0.5,
        rotate_range=(0.1), scale_range=(0.1, 0.1), translate_range=(10, 10),
        padding_mode="zeros",
    ),
    mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    mt.NormalizeIntensityd(keys=["image"], subtrahend=subtrahend, divisor=divisor),
])

val_transforms = mt.Compose([
    mt.LoadImaged(keys=["image"]),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.RepeatChanneld(keys=["image"], repeats=3),
    mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=config.img_size, mode="constant", constant_values=0),
    mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    mt.NormalizeIntensityd(keys=["image"], subtrahend=subtrahend, divisor=divisor),
])

train_dataset = CardiacMRIDataset(file_list=train_files, transform=train_transforms)
val_dataset = CardiacMRIDataset(file_list=val_files, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# --- Model, Optimizer, and Losses ---
print(f"Loading VAE from {config.model_name}")
vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae").to(config.device)

optimizer = torch.optim.AdamW(vae.parameters(), lr=config.learning_rate)
mse_criterion = nn.MSELoss()
perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg").to(config.device)

print(f"Starting training on {config.device}...")
epoch_loss_list = []

# --- Training Loop ---
for epoch in range(config.epochs):
    vae.train()
    epoch_loss, epoch_mse, epoch_perceptual = 0.0, 0.0, 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

    for batch in progress_bar:
        images = batch["image"].to(config.device)
        
        reconstructions = vae(images).sample
        
        mse_loss = mse_criterion(reconstructions, images)
        p_loss = perceptual_loss(reconstructions.float(), images.float())
        
        loss = mse_loss + config.perceptual_weight * p_loss
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_mse += mse_loss.item()
        epoch_perceptual += p_loss.item()
        progress_bar.set_postfix({
            "Total Loss": f"{loss.item():.4f}",
            "MSE": f"{mse_loss.item():.4f}",
            "Perceptual": f"{p_loss.item():.4f}"
        })
        
    avg_loss = epoch_loss / len(train_loader)
    epoch_loss_list.append(avg_loss)
    print(f"Epoch {epoch+1} Avg Total Loss: {avg_loss:.4f} "
          f"(MSE: {epoch_mse/len(train_loader):.4f}, Perceptual: {epoch_perceptual/len(train_loader):.4f})")
    
    # Validation and Saving
    if (epoch + 1) % config.val_interval == 0:
        vae.eval()
        with torch.no_grad():
            val_batch = first(val_loader)
            if val_batch is None or not val_batch: continue

            val_images = val_batch["image"].to(config.device)
            val_recons = vae(val_images).sample
            
            val_images = (val_images * 0.5 + 0.5).clamp(0, 1)
            val_recons = (val_recons * 0.5 + 0.5).clamp(0, 1)

            n_examples = val_images.shape[0]
            fig, ax = plt.subplots(2, n_examples, figsize=(n_examples * 2.5, 5.5))
            for i in range(n_examples):
                ax[0, i].imshow(val_images[i, 0].cpu().numpy(), cmap="gray")
                ax[0, i].set_title("Original"), ax[0, i].axis("off")
                ax[1, i].imshow(val_recons[i, 0].cpu().numpy(), cmap="gray")
                ax[1, i].set_title("Reconstruction"), ax[1, i].axis("off")
            
            plt.suptitle(f"Epoch {epoch+1} Validation Samples")
            plt.tight_layout()
            plt.savefig(os.path.join(config.output_dir, "samples", f"epoch_{epoch+1:03d}.png"))
            plt.close()
            print(f"Saved validation samples for epoch {epoch+1}")

        vae.save_pretrained(config.output_dir)
        print(f"Saved fine-tuned VAE to {config.output_dir}")
        
    # --- Final Save and Plot ---
    vae.save_pretrained(config.output_dir)
    print(f"Final fine-tuned VAE saved to {config.output_dir}")

    plt.figure("Learning Curve", figsize=(10, 5))
    plt.title("VAE Finetuning - Total Loss")
    plt.plot(epoch_loss_list)
    plt.xlabel("Epoch"), plt.ylabel("Weighted Loss"), plt.grid(True)
    plt.savefig(os.path.join(config.output_dir, "learning_curve.png"))
    plt.close()
    print("Saved learning curve.")