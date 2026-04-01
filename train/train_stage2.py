import os
import sys
import shutil
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusers import AutoencoderKL
from monai.losses import PerceptualLoss, PatchAdversarialLoss
from monai.networks.nets import PatchDiscriminator
from monai import transforms as mt
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.blocks import LatentFlowAndResidualPredictor
from models.losses import GradientLoss, ResidualGradientLoss, flow_smoothness_loss
from utils import prepare_pair_datalists, warp_latent, flow_to_rgb

print_config()
set_determinism(42)

# --- Configuration ---
class Config:
    dataset_root = "./data/dataset/"
    # Path to the VAE fine-tuned in Step 1
    vae_model_path = "./results/001_vae_finetuned/"
    # Directory to save the new flow model and results
    output_dir = "./results/002_lfm/"

    # --- Dataset Split Configuration ---
    acdc_train_patients = 100
    acdc_val_patients = 20
    
    dsb_train_patients = 500
    dsb_val_patients = 50
    # ----------------------------------------

    max_frame_dist = 12 # Max distance between ref and dri frames

    # Model parameters
    latent_channels = 4 # SD VAE has 4 latent channels
    img_size = (192, 192)

    # --- Loss Weights (Adjusted for Frozen VAE + Refiner) ---
    l1_weight = 1.0                 # Image L1 is now a secondary objective
    perceptual_weight = 0.5         # Perceptual loss is still useful

    warp_aux_weight = 5.0           # Give the warp a strong, direct signal. Should be comparable to latent_l1_weight.
    smoothness_weight = 2e-6        # Keep this small
    gradient_weight = 1.0           # Secondary image-space objective
    residual_l1_weight = 0.05       # Smaller regularization for the residual
    residual_gradient_weight = 0.2  # Smaller regularization
    gan_weight = 0.2                # Keep GAN weight relatively low to start
    
    # --- GAN Parameters ---
    discriminator_lr = 1e-5
    gan_loss_type = "least_squares"
    
    # --- Other Training Params ---
    steps_per_epoch = 10000          
    epochs = 30
    batch_size = 16
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_interval = 5
    flow_warmup_epochs = 5

config = Config()

# --- Create Output Directories ---
os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)

class CardiacCinePairDataset(Dataset):
    def __init__(self, pair_list, transform):
        self.pair_list, self.transform = pair_list, transform
    def __len__(self): return len(self.pair_list)
    def __getitem__(self, idx): return self.transform(self.pair_list[idx])

# --- Main Training Function ---
def train_flow_predictor():
    # 1. Load VAE and initialize Flow Predictor
    print(f"Loading VAE from: {config.vae_model_path}")
    vae = AutoencoderKL.from_pretrained(config.vae_model_path).to(config.device)
    flow_predictor = LatentFlowAndResidualPredictor(in_channels=2 * config.latent_channels, latent_channels=config.latent_channels).to(config.device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE is frozen.")

    # Initialize Discriminator
    print("Initializing PatchGAN Discriminator...")
    discriminator = PatchDiscriminator(
        spatial_dims=2, in_channels=3, num_layers_d=3, channels=64,
    ).to(config.device)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=config.discriminator_lr)

    # 2. Setup Generator optimizer and scheduler
    optimizer_g = torch.optim.AdamW([
        {'params': flow_predictor.parameters(), 'lr': config.learning_rate},
    ])
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=config.epochs, eta_min=1e-6)

    # 3. Define losses
    l1_criterion = nn.L1Loss()
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg").to(config.device)
    gradient_criterion = GradientLoss(config.device).to(config.device)
    residual_gradient_criterion = ResidualGradientLoss(config.device).to(config.device)
    gan_loss = PatchAdversarialLoss(criterion=config.gan_loss_type).to(config.device)

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    # 4. Prepare data
    train_pairs, val_pairs = prepare_pair_datalists(
        root_dir=config.dataset_root,
        acdc_train_count=config.acdc_train_patients, acdc_val_count=config.acdc_val_patients,
        dsb_train_count=config.dsb_train_patients, dsb_val_count=config.dsb_val_patients,
        max_dist=config.max_frame_dist
    )
    data_keys = ["ref_image", "dri_image"]
    train_transforms = mt.Compose([
        mt.LoadImaged(keys=data_keys), mt.EnsureChannelFirstd(keys=data_keys), mt.RepeatChanneld(keys=data_keys, repeats=3),
        mt.ResizeWithPadOrCropd(keys=data_keys, spatial_size=config.img_size, mode="constant", constant_values=0),
        mt.RandAffined(keys=data_keys, prob=0.8, rotate_range=(0.1), scale_range=(0.1, 0.1), translate_range=(10, 10), padding_mode="zeros"),
        mt.ScaleIntensityRanged(keys=data_keys, a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
    ])
    val_transforms = mt.Compose([
        mt.LoadImaged(keys=data_keys), mt.EnsureChannelFirstd(keys=data_keys), mt.RepeatChanneld(keys=data_keys, repeats=3),
        mt.ResizeWithPadOrCropd(keys=data_keys, spatial_size=config.img_size, mode="constant", constant_values=0),
        mt.ScaleIntensityRanged(keys=data_keys, a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
    ])
    train_ds = CardiacCinePairDataset(train_pairs, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True) # Increased workers
    val_loader = DataLoader(CardiacCinePairDataset(val_pairs, val_transforms), batch_size=4, shuffle=True, num_workers=4)

    # 5. Training loop
    print(f"Starting latent flow training on {config.device}...")
    best_val_loss = float('inf')
    epoch_losses_history = defaultdict(list)
    
    # ---Main training loop ---
    train_iterator = iter(train_loader)
    
    for epoch in range(config.epochs):
        is_flow_warmup = epoch < config.flow_warmup_epochs

        flow_predictor.train()
        discriminator.train(not is_flow_warmup)
        
        batch_losses = defaultdict(float)
        desc = f"Epoch {epoch+1}/{config.epochs}"
        if is_flow_warmup:
            desc += " (Flow Warm-up)"
            
        progress_bar = tqdm(range(config.steps_per_epoch), desc=desc)

        for step in progress_bar:
            try:
                batch = next(train_iterator)
            except StopIteration:
                # DataLoader is exhausted, create a new iterator to loop over the data
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            ref_imgs = batch["ref_image"].to(config.device)
            dri_imgs = batch["dri_image"].to(config.device)

            # === Generator Update ===
            optimizer_g.zero_grad(set_to_none=True) # Use set_to_none=True for a small performance gain
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z_ref = vae.encode(ref_imgs).latent_dist.sample()
                    z_dri = vae.encode(dri_imgs).latent_dist.sample()

                # --- Forward Pass ---
                latent_flow, latent_residual = flow_predictor(z_ref, z_dri)
                z_warped = warp_latent(z_ref, latent_flow)

                if is_flow_warmup:
                    loss_latent_warp = l1_criterion(z_warped + latent_residual, z_dri)
                    loss_s = flow_smoothness_loss(latent_flow)
                    total_g_loss = loss_latent_warp + config.smoothness_weight * loss_s
                    
                    batch_losses['G_Total_Warmup'] += total_g_loss.item()
                    batch_losses['Latent_Warp'] += loss_latent_warp.item()
                    progress_bar.set_postfix({"G_Warmup": f"{total_g_loss.item():.3f}", "L_Warp": f"{loss_latent_warp.item():.4f}"})
                
                else:
                    # --- FULL TRAINING PHASE with AUXILIARY LOSS ---
                    z_pred = z_warped + latent_residual
                    reconstruction_dri = vae.decode(z_pred).sample

                    loss_latent_warp_aux = l1_criterion(z_pred, z_dri)
                    loss_l1 = l1_criterion(reconstruction_dri, dri_imgs)
                    loss_p = perceptual_loss(reconstruction_dri, dri_imgs)
                    loss_grad = gradient_criterion(reconstruction_dri, dri_imgs)
                    loss_s = flow_smoothness_loss(latent_flow)
                    loss_residual_reg = latent_residual.abs().mean()
                    loss_res_grad = residual_gradient_criterion(latent_residual)
                    
                    # We need to run the discriminator in the autocast context for the G loss
                    d_fake_output = discriminator(reconstruction_dri)
                    loss_g = gan_loss(d_fake_output, target_is_real=True, for_discriminator=False)

                    total_g_loss = (
                        config.warp_aux_weight * loss_latent_warp_aux +
                        config.l1_weight * loss_l1 +
                        config.perceptual_weight * loss_p +
                        config.gradient_weight * loss_grad +
                        config.smoothness_weight * loss_s +
                        config.residual_l1_weight * loss_residual_reg +
                        config.residual_gradient_weight * loss_res_grad +
                        config.gan_weight * loss_g
                    )
                    
                    batch_losses['G_Total'] += total_g_loss.item()
                    batch_losses['Latent_Warp_Aux'] += loss_latent_warp_aux.item()
                    batch_losses['L1_Image'] += loss_l1.item()
                    batch_losses['GAN_G'] += loss_g.item()
                    progress_bar.set_postfix({
                        "G": f"{total_g_loss.item():.2f}", "D": f"{batch_losses['D_Total'] / (step + 1):.2f}",
                        "L1_Img": f"{loss_l1.item():.3f}"
                    })

            # --- Scale loss and step optimizer ---
            scaler_g.scale(total_g_loss).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # === Discriminator Update (only after warm-up) ===
            if not is_flow_warmup:
                optimizer_d.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    loss_d_real = gan_loss(discriminator(dri_imgs), target_is_real=True, for_discriminator=True)
                    loss_d_fake = gan_loss(discriminator(reconstruction_dri.detach()), target_is_real=False, for_discriminator=True)
                    total_d_loss = (loss_d_real + loss_d_fake) * 0.5

                # --- Scale loss and step optimizer ---
                scaler_d.scale(total_d_loss).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
                
                batch_losses['D_Total'] += total_d_loss.item()

        # --- End of Epoch Logging ---
        num_steps_in_epoch = config.steps_per_epoch
        for key in batch_losses:
            epoch_losses_history[key].append(batch_losses[key] / num_steps_in_epoch)
        
        scheduler_g.step()
        current_lr = scheduler_g.get_last_lr()[0]
        
        if is_flow_warmup:
            print(f"Epoch {epoch+1} Avg G_Warmup_Loss: {epoch_losses_history['G_Total_Warmup'][-1]:.4f}, LR: {current_lr:.2e}")
        else:
            if epoch == config.flow_warmup_epochs:
                 print("\n" + "="*50 + "\nFlow warm-up complete. Starting full training.\n" + "="*50)
            print(f"Epoch {epoch+1} Avg G_Loss: {epoch_losses_history['G_Total'][-1]:.4f}, Avg D_Loss: {epoch_losses_history['D_Total'][-1]:.4f}, LR: {current_lr:.2e}")

        # 6. Validation and visualization
        if (epoch + 1) % config.val_interval == 0 or epoch == 0 or epoch == config.flow_warmup_epochs:
            flow_predictor.eval()
            val_latent_loss = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    ref_val, dri_val = val_batch["ref_image"].to(config.device), val_batch["dri_image"].to(config.device)
                    
                    z_ref_val = vae.encode(ref_val).latent_dist.sample()
                    z_dri_val = vae.encode(dri_val).latent_dist.sample()
                    
                    flow_val, residual_val = flow_predictor(z_ref_val, z_dri_val)
                    z_pred_val = warp_latent(z_ref_val, flow_val) + residual_val
                    
                    # The validation loss is the L1 distance between the refined latent and the ground truth latent
                    val_latent_loss += l1_criterion(z_pred_val, z_dri_val).item()

                avg_val_latent_loss = val_latent_loss / len(val_loader)
                print(f"Validation Latent L1 Loss: {avg_val_latent_loss:.4f}")

                if avg_val_latent_loss < best_val_loss:
                    best_val_loss = avg_val_latent_loss
                    print(f"New best model! Saving checkpoints with val loss {best_val_loss:.4f}")
                    torch.save(flow_predictor.state_dict(), os.path.join(config.output_dir, "checkpoints", "best_flow_predictor.pth"))
                    torch.save(discriminator.state_dict(), os.path.join(config.output_dir, "checkpoints", "best_discriminator.pth"))

                # --- Visualization now matches the new pipeline ---
                val_batch_first = first(val_loader)
                ref_val, dri_val = val_batch_first["ref_image"].to(config.device), val_batch_first["dri_image"].to(config.device)

                z_ref_val = vae.encode(ref_val).latent_dist.sample()
                z_dri_val = vae.encode(dri_val).latent_dist.sample()

                flow_val, residual_val = flow_predictor(z_ref_val, z_dri_val)
                z_pred_val = warp_latent(z_ref_val, flow_val) + residual_val

                recon_dri_val = vae.decode(z_pred_val).sample

                n_examples = recon_dri_val.shape[0]
                fig, ax = plt.subplots(8, n_examples, figsize=(n_examples * 3, 12))
                for i in range(n_examples):
                    ref_disp = (ref_val[i, 0] * 0.5 + 0.5).clamp(0, 1).cpu()
                    dri_disp = (dri_val[i, 0] * 0.5 + 0.5).clamp(0, 1).cpu()
                    recon_disp = (recon_dri_val[i, 0] * 0.5 + 0.5).clamp(0, 1).cpu()
                    ax[0, i].imshow(ref_disp, cmap="gray"); ax[0, i].set_title("Reference")
                    ax[1, i].imshow(dri_disp, cmap="gray"); ax[1, i].set_title("Driving (GT)")
                    ax[2, i].imshow(recon_disp, cmap="gray"); ax[2, i].set_title("Generated")
                    max_flow_val = torch.abs(flow_val[i]).max().item() + 1e-6
                    ax[3, i].imshow(flow_val[i, 0].cpu().numpy(), cmap="coolwarm", vmin=-max_flow_val, vmax=max_flow_val)
                    if i == 0: ax[3, i].set_ylabel("Flow X")
                    ax[4, i].imshow(flow_val[i, 1].cpu().numpy(), cmap="coolwarm", vmin=-max_flow_val, vmax=max_flow_val)
                    if i == 0: ax[4, i].set_ylabel("Flow Y")
                    ax[5, i].imshow(flow_to_rgb(flow_val[i])); ax[5, i].set_title("Latent Flow")
                    res_disp = residual_val[i, 0].cpu().numpy()
                    max_res_val = np.abs(res_disp).max() + 1e-6
                    im = ax[6, i].imshow(res_disp, cmap="coolwarm", vmin=-max_res_val, vmax=max_res_val)
                    ax[6, i].set_title("Residual (Ch 0)"); fig.colorbar(im, ax=ax[6, i])
                    ax[7, i].imshow(dri_disp-recon_disp, cmap="coolwarm"); ax[7, i].set_title("Error Map")
                    for r in range(8): ax[r, i].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(config.output_dir, "samples", f"epoch_{epoch+1:03d}.png"))
                plt.close()
            
            plt.figure("Loss Components", figsize=(12, 8))
            plt.title("LFM Training - Loss Components")
            for key, value in epoch_losses_history.items():
                if np.mean(value) < 1e-5: continue # Increased threshold slightly
                plt.plot(value, label=key)
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.ylim(bottom=0)
            plt.savefig(os.path.join(config.output_dir, "loss_components_curve.png"))
            plt.close()
            print("Saved validation samples and loss curve.")

if __name__ == "__main__":
    try:
        shutil.copy(sys.argv[0], os.path.join(config.output_dir, os.path.basename(sys.argv[0])))
        print(f"Copied training script to '{config.output_dir}'")
    except Exception as e:
        print(f"Warning: Could not copy training script. Error: {e}")
    train_flow_predictor()