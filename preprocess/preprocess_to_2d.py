import os
import nibabel as nib
import numpy as np
import imageio
from tqdm import tqdm

def normalize_to_uint8(data):
    """
    Normalizes a numpy array to the 0-255 range and converts to uint8.
    
    Args:
        data (np.array): Input numpy array.
        
    Returns:
        np.array: Normalized uint8 array.
    """
    # Check for empty or all-zero arrays
    if np.max(data) == np.min(data):
        return np.zeros(data.shape, dtype=np.uint8)
        
    # Min-max normalization
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Scale to 0-255 and convert to uint8
    return (data * 255).astype(np.uint8)

def extract_and_save_frames(nifti_path, dataset_name, patient_id, base_output_dir):
    """
    Loads a 4D NIfTI file, extracts each 2D frame (slice-time pair),
    and saves it as a PNG image in a structured directory.
    
    Args:
        nifti_path (str): Path to the 4D .nii.gz file.
        dataset_name (str): Name of the dataset (e.g., 'ACDC', 'DSB').
        patient_id (str): The identifier for the patient.
        base_output_dir (str): The root directory to save the processed frames.
    """
    try:
        # Load the NIfTI file
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()

        # Ensure data is 4D
        if len(data.shape) != 4:
            print(f"  [Skipping] Expected 4D data, but got {data.shape} for {nifti_path}")
            return
            
        # Get dimensions (width, height, slices, time)
        _, _, num_slices, num_frames = data.shape
        
        # Normalize the entire 4D volume at once to maintain consistent intensity
        normalized_data = normalize_to_uint8(data)

        # Loop through each slice (Z dimension)
        for z in range(num_slices):
            # Create the specific output folder for this slice
            # Format: DATASETNAME_patientName_sliceN
            slice_folder_name = f"{dataset_name}_{patient_id}_slice{z}"
            slice_output_path = os.path.join(base_output_dir, slice_folder_name)
            os.makedirs(slice_output_path, exist_ok=True)
            
            # Loop through each time frame (t dimension)
            for t in range(num_frames):
                # Extract the 2D frame
                frame_data = normalized_data[:, :, z, t]
                
                # The MRI data might be stored in a rotated way.
                # We can rotate and flip it to have a standard orientation (e.g., superior-anterior).
                # frame_data = np.rot90(frame_data)
                
                # Define the output filename
                # Using z-fill to ensure correct sorting (e.g., frame_001.png, frame_002.png)
                frame_filename = f"frame_{t:03d}.png"
                frame_output_path = os.path.join(slice_output_path, frame_filename)
                
                # Save the frame as a PNG image
                imageio.imwrite(frame_output_path, frame_data)
                
    except Exception as e:
        print(f"  [Error] Failed to process {nifti_path}: {e}")


def process_datasets(base_data_dir, output_dir):
    """
    Main function to find and process all relevant NIfTI files from the datasets.
    """
    # Create the main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")
    
    datasets_info = {
        'ACDC': 'ACDC_Preprocessed',
        'DSB': 'DSB_nifti'
    }

    for dataset_name, dataset_folder in datasets_info.items():
        dataset_path = os.path.join(base_data_dir, dataset_folder)
        
        if not os.path.isdir(dataset_path):
            print(f"Dataset directory not found: {dataset_path}")
            continue
            
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Iterate through train/test/val splits
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.isdir(split_path):
                continue

            # Get a list of patient directories
            patient_dirs = sorted([p for p in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, p))])
            
            if not patient_dirs:
                continue

            print(f"Found {len(patient_dirs)} patients in '{split}' split.")
            
            # Use tqdm for a progress bar
            for patient_id in tqdm(patient_dirs, desc=f'  -> {split.capitalize()}'):
                patient_path = os.path.join(split_path, patient_id)
                
                # Find the 4D cine MRI file (_sax_t.nii.gz)
                target_file = None
                for filename in os.listdir(patient_path):
                    if 'sax_t.nii.gz' in filename:
                        target_file = filename
                        break
                
                if target_file:
                    nifti_path = os.path.join(patient_path, target_file)
                    extract_and_save_frames(nifti_path, dataset_name, patient_id, output_dir)
                else:
                    print(f"  [Warning] No '_sax_t.nii.gz' file found for patient {patient_id} in {split_path}")


if __name__ == "__main__":
    # --- Configuration ---
    # The root folder containing 'ACDC_Preprocessed' and 'DSB_nifti'
    BASE_DATA_FOLDER = './data/' 
    # The folder where all the 2D frames will be saved
    OUTPUT_2D_FOLDER = './data/dataset'
    
    process_datasets(BASE_DATA_FOLDER, OUTPUT_2D_FOLDER)
    
    print("\nPreprocessing complete!")