import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

def find_es_frame_from_segmentation(seg_filepath):
    """
    Calculates the volume for each time frame in a 4D segmentation NIfTI file
    and returns the frame index with the minimum volume (ES frame).

    Args:
        seg_filepath (str): The full path to the 4D segmentation NIfTI file.

    Returns:
        int: The index of the End-Systolic (ES) frame. Returns -1 if an error occurs.
    """
    try:
        # Load the NIfTI file
        seg_nii = nib.load(seg_filepath)
        seg_data = seg_nii.get_fdata()

        # Ensure the data is 4D (x, y, z, time)
        if seg_data.ndim != 4:
            print(f"Warning: Expected 4D data, but got {seg_data.ndim}D for {os.path.basename(seg_filepath)}. Skipping.")
            return -1

        # Get the number of time frames
        num_frames = seg_data.shape[3]
        
        # We don't need real-world volume (in mm^3) to find the minimum,
        # just the voxel count is sufficient and faster.
        # But if you wanted real volume, you would do this:
        # voxel_dims = seg_nii.header.get_zooms()
        # voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]

        volumes = []
        for t in range(num_frames):
            # Get the 3D segmentation mask for the current time frame
            frame_data = seg_data[:, :, :, t]
            
            # Count the number of non-zero voxels (this is proportional to volume)
            num_voxels = np.count_nonzero(frame_data)
            volumes.append(num_voxels)

        # Find the index of the minimum volume
        # This index is our estimated ES frame
        if not volumes:
            return -1 # Handle cases with no segmented voxels
            
        es_frame = np.argmin(volumes)
        return int(es_frame)

    except Exception as e:
        print(f"Error processing {os.path.basename(seg_filepath)}: {e}")
        return -1


# --- Main Script ---

segmentation_dir = './data/DSB_nifti/' 

# Path for the output CSV file
output_csv_path = 'es_frames.csv'

# --- Script Execution ---

if not os.path.isdir(segmentation_dir):
    print(f"Error: The directory '{segmentation_dir}' does not exist.")
    print("Please update the 'segmentation_dir' variable in the script.")
else:
    # Find all segmentation files in the directory
    seg_files = [f for f in os.listdir(segmentation_dir) if f.endswith('_segmentation.nii.gz')]

    results = []

    print(f"Found {len(seg_files)} segmentation files. Starting processing...")

    # Use tqdm for a progress bar
    for filename in tqdm(seg_files, desc="Analyzing Volumes"):
        # Extract patient ID from filename (e.g., '1_sax_nii_segmentation.nii.gz' -> '1')
        try:
            patient_id = int(filename.split('_')[0])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse patient ID from filename '{filename}'. Skipping.")
            continue

        # Get the full path to the file
        file_path = os.path.join(segmentation_dir, filename)

        # Find the ES frame
        es_frame_index = find_es_frame_from_segmentation(file_path)

        # Store the result
        if es_frame_index != -1:
            results.append({
                'pid': patient_id,
                'es_frame': es_frame_index
            })

    # Convert results to a pandas DataFrame and save to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='pid').reset_index(drop=True) # Sort by patient ID
        
        results_df.to_csv(output_csv_path, index=False)
        
        print("\n--- Processing Complete ---")
        print(f"Successfully saved ES frame information to: {output_csv_path}")
        print("\n--- Sample Output ---")
        print(results_df.head())
    else:
        print("No valid segmentation files were processed.")