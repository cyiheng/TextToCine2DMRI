import os
import re
import random
import glob
from collections import defaultdict

import numpy as np
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F

def prepare_datalists(root_dir, acdc_train_count, acdc_val_count, dsb_train_count, dsb_val_count):
    """
    Prepares patient-aware train/validation splits from multiple datasets (ACDC, DSB).
    """
    print("--- Preparing Patient-Aware Datalists for ACDC & DSB ---")
    
    # This dictionary will store file paths grouped by a unique patient tuple: (dataset_name, patient_id)
    # e.g., ('ACDC', 101) -> ['path/to/img1.png', 'path/to/img2.png']
    patient_to_files = defaultdict(list)
    
    # Regex to capture dataset name and patient ID from the folder structure
    # It matches "ACDC_patient123_..." or "DSB_456_..."
    pattern = re.compile(r"^(ACDC)_patient(\d+)|^(DSB)_(\d+)")

    all_slice_folders = glob.glob(os.path.join(root_dir, "*"))

    print(f"Scanning {len(all_slice_folders)} slice folders...")
    for folder_path in all_slice_folders:
        folder_name = os.path.basename(folder_path)
        match = pattern.match(folder_name)
        
        if match:
            # The regex has two capture groups for each case (ACDC or DSB)
            # e.g., ('ACDC', '123', None, None) or (None, None, 'DSB', '456')
            acdc_name, acdc_id, dsb_name, dsb_id = match.groups()
            
            if acdc_name:
                dataset = acdc_name
                patient_id = int(acdc_id)
            else:
                dataset = dsb_name
                patient_id = int(dsb_id)

            unique_patient_key = (dataset, patient_id)
            
            # Add all png files from this folder to the patient's list
            png_files = glob.glob(os.path.join(folder_path, "*.png"))
            patient_to_files[unique_patient_key].extend(png_files)

    # Separate patient keys by dataset
    acdc_patients = sorted([key for key in patient_to_files if key[0] == 'ACDC'], key=lambda x: x[1])
    dsb_patients = sorted([key for key in patient_to_files if key[0] == 'DSB'], key=lambda x: x[1])
    
    print(f"Found {len(acdc_patients)} unique ACDC patients.")
    print(f"Found {len(dsb_patients)} unique DSB patients.")

    # --- Select Patients for Training and Validation ---
    # ACDC split
    acdc_train_pids = acdc_patients[:acdc_train_count]
    acdc_val_pids = acdc_patients[acdc_train_count : acdc_train_count + acdc_val_count]
    
    # DSB split
    dsb_train_pids = dsb_patients[:dsb_train_count]
    dsb_val_pids = dsb_patients[dsb_train_count : dsb_train_count + dsb_val_count]
    
    # Combine patient lists
    train_pids = acdc_train_pids + dsb_train_pids
    val_pids = acdc_val_pids + dsb_val_pids
    
    # --- Collect all files for the selected patients ---
    train_files = []
    for pid in train_pids:
        train_files.extend(patient_to_files[pid])
        
    val_files = []
    for pid in val_pids:
        val_files.extend(patient_to_files[pid])
        
    print("-" * 55)
    print(f"Total Training Patients: {len(train_pids)} ({len(acdc_train_pids)} ACDC + {len(dsb_train_pids)} DSB)")
    print(f"Total Training Images:   {len(train_files)}")
    print(f"Total Validation Patients: {len(val_pids)} ({len(acdc_val_pids)} ACDC + {len(dsb_val_pids)} DSB)")
    print(f"Total Validation Images:   {len(val_files)}")
    print("-" * 55)
    
    return train_files, val_files


def prepare_pair_datalists(
    root_dir, 
    acdc_train_count, 
    acdc_val_count, 
    dsb_train_count, 
    dsb_val_count, 
    max_dist
):
    """
    Prepares patient-aware train/validation splits of image pairs from multiple datasets.
    """
    print("--- Preparing Patient-Aware Paired Datalists for ACDC & DSB ---")
    
    # This dictionary will store frame info grouped by:
    # (dataset, patient_id) -> slice_directory -> [list of frames]
    patient_to_slices = defaultdict(lambda: defaultdict(list))
    
    # Regex to capture dataset name and patient ID from the folder structure
    pattern = re.compile(r"^(ACDC)_patient(\d+)|^(DSB)_(\d+)")

    all_png_files = glob.glob(os.path.join(root_dir, "*", "*.png"))
    print(f"Scanning {len(all_png_files)} total image files...")

    for f_path in all_png_files:
        try:
            slice_dir = os.path.basename(os.path.dirname(f_path))
            match = pattern.match(slice_dir)
            frame_match = re.search(r'frame_(\d+)', os.path.basename(f_path))

            if match and frame_match:
                # Extract dataset and patient ID
                acdc_name, acdc_id, dsb_name, dsb_id = match.groups()
                if acdc_name:
                    dataset, patient_id = acdc_name, int(acdc_id)
                else:
                    dataset, patient_id = dsb_name, int(dsb_id)

                unique_patient_key = (dataset, patient_id)
                frame_idx = int(frame_match.group(1))
                
                patient_to_slices[unique_patient_key][slice_dir].append({"path": f_path, "idx": frame_idx})
        except (ValueError, TypeError):
            # Catch potential errors if filenames are not as expected
            continue
            
    # --- Patient Splitting Logic ---
    all_patients = patient_to_slices.keys()
    acdc_patients = sorted([key for key in all_patients if key[0] == 'ACDC'], key=lambda x: x[1])
    dsb_patients = sorted([key for key in all_patients if key[0] == 'DSB'], key=lambda x: x[1])

    print(f"Found {len(acdc_patients)} unique ACDC patients.")
    print(f"Found {len(dsb_patients)} unique DSB patients.")

    # Select patients from each dataset
    acdc_train_pids = acdc_patients[:acdc_train_count]
    acdc_val_pids = acdc_patients[acdc_train_count : acdc_train_count + acdc_val_count]
    
    dsb_train_pids = dsb_patients[:dsb_train_count]
    dsb_val_pids = dsb_patients[dsb_train_count : dsb_train_count + dsb_val_count]
    
    # Combine into sets for efficient lookup during pair creation
    train_pids = set(acdc_train_pids + dsb_train_pids)
    val_pids = set(acdc_val_pids + dsb_val_pids)
    
    # --- Pair Creation ---
    train_pairs, val_pairs = [], []
    
    # Iterate through all found patients
    for patient_key, slices in patient_to_slices.items():
        # For each slice directory belonging to that patient
        for slice_dir, frames in slices.items():
            if len(frames) < 2:
                continue
            
            # Sort frames by their index (frame_001, frame_002, etc.)
            frames.sort(key=lambda x: x['idx'])
            num_frames = len(frames)
            
            # Create pairs
            for i in range(num_frames):
                # The maximum forward distance to the second frame in the pair
                max_k = min(max_dist, num_frames - 1 - i)
                
                if max_k > 0:
                    # Choose a random distance
                    k = random.randint(1, max_k)
                    ref_frame, dri_frame = frames[i], frames[i + k]
                    
                    pair = {"ref_image": ref_frame["path"], "dri_image": dri_frame["path"]}
                    
                    # Assign the pair to the correct list based on the patient key
                    if patient_key in train_pids:
                        train_pairs.append(pair)
                    elif patient_key in val_pids:
                        val_pairs.append(pair)

    print("-" * 55)
    print(f"Total Training Patients: {len(train_pids)} ({len(acdc_train_pids)} ACDC + {len(dsb_train_pids)} DSB)")
    print(f"Total Training Pairs:    {len(train_pairs)}")
    print(f"Total Validation Patients: {len(val_pids)} ({len(acdc_val_pids)} ACDC + {len(dsb_val_pids)} DSB)")
    print(f"Total Validation Pairs:    {len(val_pairs)}")
    print("-" * 55)
    
    return train_pairs, val_pairs



def warp_latent(z_ref, latent_flow):
    """Warps a latent tensor using a flow field."""
    b, _, h, w = z_ref.shape
    xx, yy = torch.meshgrid(torch.arange(w, device=z_ref.device), torch.arange(h, device=z_ref.device), indexing='xy')
    grid = torch.stack([xx, yy], dim=0).float()
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    vgrid = grid + latent_flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    z_warped = F.grid_sample(z_ref, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return z_warped


def flow_to_rgb(flow):
    """Converts a flow field to an RGB image for visualization."""
    # This function expects a 2-channel flow field (C, H, W)
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
        
    # Squeeze out batch dimension if it exists
    if flow.ndim == 4:
        flow = flow[0]
        
    fx, fy = flow[0, :, :], flow[1, :, :]
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    
    # Create HSV image
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue from angle
    hsv[..., 1] = 1.0                           # Saturation is always max
    
    # Normalize magnitude for Value channel
    if magnitude.max() > 0:
        # Robust normalization using percentile to handle outliers
        v_norm = np.clip(magnitude / np.percentile(magnitude, 99.5), 0, 1)
        hsv[..., 2] = v_norm
        
    # Convert HSV to RGB
    rgb = mcolors.hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)
