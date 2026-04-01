import os
import sys
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL

from monai import transforms as mt
from monai.config import print_config

from utils import prepare_datalists

print_config()

# --- Configuration ---
class Config:
    # Path to the directory where your fine-tuned VAE is saved
    model_path = "./results/001_vae_finetuned/"
    
    # Path to your dataset
    dataset_root = "./data/dataset/"
    
    # Path where the output comparison images will be saved
    output_dir = os.path.join(model_path, "inference_results/")
    
    # Use the first 100 patients from ACDC for training
    acdc_train_patients = 100
    # Use the next 20 patients from ACDC for validation
    acdc_val_patients = 20
    
    # Use the first 500 patients from DSB for training
    dsb_train_patients = 500
    # Use the next 50 patients from DSB for validation
    dsb_val_patients = 50
    
    # Model and data parameters
    img_size = (192,192)
    batch_size = 4 # Can be smaller or larger depending on your VRAM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# --- Create Output Directory ---
os.makedirs(config.output_dir, exist_ok=True)
print(f"Results will be saved in: {config.output_dir}")

# --- Dataset Class (Copied from training script) ---
class CardiacMRIDataset(Dataset):
    def __init__(self, file_list, transform):
        self.image_paths = file_list
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        data = {"image": img_path, "path": img_path} # Also return path for reference
        return self.transform(data)

# --- Main Inference Logic ---
def run_inference():
    # 1. Load the fine-tuned VAE
    print(f"Loading fine-tuned VAE from {config.model_path}")
    vae = AutoencoderKL.from_pretrained(config.model_path).to(config.device)
    vae.eval() # Set the model to evaluation mode
    print("Model loaded successfully.")

    # 2. Prepare the validation dataset
    _, val_files = prepare_datalists(
        root_dir=config.dataset_root,
        acdc_train_count=config.acdc_train_patients,
        acdc_val_count=config.acdc_val_patients,
        dsb_train_count=config.dsb_train_patients,
        dsb_val_count=config.dsb_val_patients
    )

    # Define normalization values once
    subtrahend = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    divisor = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    # Use the exact same transforms as validation during training (no augmentations)
    val_transforms = mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.RepeatChanneld(keys=["image"], repeats=3),
        mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=config.img_size, mode="constant", constant_values=0),
        mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        mt.NormalizeIntensityd(keys=["image"], subtrahend=subtrahend, divisor=divisor),
    ])

    val_dataset = CardiacMRIDataset(file_list=val_files, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 3. Run the inference loop
    print("\nStarting inference on the validation set...")
    with torch.no_grad(): # Disable gradient calculations for efficiency
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Reconstructing")
        for i, batch in progress_bar:
            images = batch["image"].to(config.device)
            
            # Get reconstructions from the VAE
            reconstructions = vae(images).sample
            
            # De-normalize images from [-1, 1] to [0, 1] for visualization
            images = (images * 0.5 + 0.5).clamp(0, 1)
            reconstructions = (reconstructions * 0.5 + 0.5).clamp(0, 1)

            # 4. Save the comparison images
            n_examples = images.shape[0]
            fig, ax = plt.subplots(2, n_examples, figsize=(n_examples * 2.5, 5.5))
            
            # Handle the case where there is only one image in the batch
            if n_examples == 1:
                ax = ax[:, None]

            for j in range(n_examples):
                original_img = images[j, 0].cpu().numpy()
                recon_img = reconstructions[j, 0].cpu().numpy()
                
                ax[0, j].imshow(original_img, cmap="gray")
                ax[0, j].set_title("Original")
                ax[0, j].axis("off")
                
                ax[1, j].imshow(recon_img, cmap="gray")
                ax[1, j].set_title("Reconstruction")
                ax[1, j].axis("off")
            
            plt.tight_layout()
            output_path = os.path.join(config.output_dir, f"reconstruction_batch_{i+1:03d}.png")
            plt.savefig(output_path, dpi=150)
            plt.close()

    print(f"\nInference complete. All comparison images saved to {config.output_dir}")

if __name__ == "__main__":
    try:
        source_script_path = sys.argv[0]
        script_name = os.path.basename(source_script_path)
        dest_script_path = os.path.join(config.model_path, script_name)
        shutil.copy(source_script_path, dest_script_path)
        print(f"Copied training script '{source_script_path}' to '{dest_script_path}'")
    except Exception as e:
        print(f"Warning: Could not copy training script. Error: {e}")
    run_inference()