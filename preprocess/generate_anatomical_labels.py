import os
import json
import pandas as pd
import re
from tqdm import tqdm

def load_all_metadata(base_data_dir):
    """
    Loads and combines metadata from all train/test/val CSV files
    for both ACDC and DSB datasets.
    """
    print("Loading metadata from CSV files...")
    all_metadata = {}
    
    datasets_info = {
        'ACDC': {
            'folder': 'ACDC_Preprocessed',
            'patient_col': 'pid'
        },
        'DSB': {
            'folder': 'DSB_nifti',
            'patient_col': 'pid'
        }
    }
    
    for dataset_name, info in datasets_info.items():
        dataset_path = os.path.join(base_data_dir, info['folder'])
        if not os.path.isdir(dataset_path):
            continue
            
        for split in ['train', 'test', 'val']:
            csv_path = os.path.join(dataset_path, f"{split}_metadata.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if info['patient_col'] in df.columns and 'n_slices' in df.columns:
                        for _, row in df.iterrows():
                            patient_id_raw = row[info['patient_col']]
                            n_slices = int(row['n_slices'])
                            # Normalize patient ID to remove leading zeros, etc.
                            try:
                                patient_id_normalized = str(int(float(patient_id_raw)))
                            except (ValueError, TypeError):
                                patient_id_normalized = str(patient_id_raw).strip()
                            
                            unique_key = f"{dataset_name}_{patient_id_normalized}"
                            all_metadata[unique_key] = n_slices
                    else:
                        print(f"  [Warning] CSV '{csv_path}' is missing '{info['patient_col']}' or 'n_slices' column.")
                except Exception as e:
                    print(f"  [Error] Could not process {csv_path}: {e}")

    print(f"Successfully loaded metadata for {len(all_metadata)} patients.")
    return all_metadata

def get_anatomical_level(slice_index, total_slices):
    """
    Determines a more granular anatomical level of a slice based on its relative
    position in the heart stack, dividing it into six parts.
    
    Args:
        slice_index (int): The 0-based index of the slice.
        total_slices (int): The total number of slices for that patient.
        
    Returns:
        tuple: (level_name, justification_text)
    """
    if total_slices <= 1:
        return "Mid-ventricular", "Only one slice available; assumed to be mid-ventricular."

    # Special case for the absolute last slice to identify the "Apical Tip"
    if slice_index == total_slices - 1:
        level = "Apical Tip"
        justification = (f"This is the final slice ({slice_index} of {total_slices-1}), representing the true apex "
                         f"where the ventricular cavity disappears.")
        return level, justification

    # Calculate the relative position using the "rule of sixths"
    relative_position = (slice_index + 0.5) / total_slices
    
    if relative_position < 1/6:
        level = "Outflow Tract / Atrial"
        justification = (f"Slice {slice_index} of {total_slices-1} is in the top sixth of the stack, likely showing "
                         f"the great vessels (aorta/pulmonary artery) or atria.")
    elif relative_position < 2/6:
        level = "Basal"
        justification = (f"Slice {slice_index} of {total_slices-1} is in the upper-basal region (second sixth), "
                         f"where the mitral valve is typically visible.")
    elif relative_position < 3/6:
        level = "Upper Mid-ventricular"
        justification = (f"Slice {slice_index} of {total_slices-1} is in the upper-mid region (third sixth), "
                         f"where papillary muscles are usually well-defined and separate.")
    elif relative_position < 4/6:
        level = "Lower Mid-ventricular"
        justification = (f"Slice {slice_index} of {total_slices-1} is in the lower-mid region (fourth sixth), "
                         f"where papillary muscles may begin to fuse with the ventricular wall.")
    else: # This covers the remaining part of the stack before the tip
        level = "Apical"
        justification = (f"Slice {slice_index} of {total_slices-1} is in the lower portion of the stack (beyond 2/3), "
                         f"where the ventricular cavity is small, approaching the apex.")
        
    return level, justification


def generate_anatomical_json(dataset_dir, metadata, output_file):
    """
    Generates a JSON file with anatomical labels for each slice folder.
    """
    if not os.path.isdir(dataset_dir):
        print(f"Error: The processed dataset directory '{dataset_dir}' was not found.")
        print("Please run the first script to generate the 2D images first.")
        return

    anatomical_data = {}
    folder_pattern = re.compile(r"^(ACDC|DSB)_(.+)_slice(\d+)$")
    
    slice_folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])

    print(f"\nFound {len(slice_folders)} slice folders to process in '{dataset_dir}'...")

    for folder_name in tqdm(slice_folders, desc="Generating Labels"):
        match = folder_pattern.match(folder_name)
        
        if not match:
            print(f"  [Warning] Skipping folder with unexpected name format: {folder_name}")
            continue
            
        dataset, patient_id, slice_num_str = match.groups()
        slice_index = int(slice_num_str)
        patient_key = f"{dataset}_{patient_id}"
        total_slices = metadata.get(patient_key)
        
        if total_slices is None:
            print(f"  [Warning] Could not find metadata for patient key: {patient_key}. Skipping {folder_name}.")
            continue
        
        level, justification = get_anatomical_level(slice_index, total_slices)
        
        anatomical_data[folder_name] = {
            "level_full_text": level,
            "level_derived": level,
            # "justification": justification
        }
        
    print(f"\nSaving data for {len(anatomical_data)} slices to '{output_file}'...")
    with open(output_file, 'w') as f:
        json.dump(anatomical_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    BASE_DATA_FOLDER = './data'
    PROCESSED_DATASET_FOLDER = './data/dataset'
    OUTPUT_JSON_FILE = './data/anatomical_levels.json'
    
    patient_metadata = load_all_metadata(BASE_DATA_FOLDER)
    
    if patient_metadata:
        generate_anatomical_json(PROCESSED_DATASET_FOLDER, patient_metadata, OUTPUT_JSON_FILE)