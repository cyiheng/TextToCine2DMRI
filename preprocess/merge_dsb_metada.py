import os
import pandas as pd
import numpy as np

def add_demographics_to_metadata(base_data_dir, dsb_folder_name, demographics_filename):
    """
    Merges Sex and Age data from a source CSV into the train, test, and val
    metadata CSVs for the DSB dataset.

    Args:
        base_data_dir (str): Path to the '000_Data' directory.
        dsb_folder_name (str): Name of the DSB nifti folder (e.g., 'DSB_nifti').
        demographics_filename (str): Filename of the CSV with Sex and Age.
    """
    
    dsb_path = os.path.join(base_data_dir, dsb_folder_name)
    demographics_path = os.path.join(dsb_path, demographics_filename)
    
    # --- 1. Safety Checks ---
    if not os.path.isdir(dsb_path):
        print(f"Error: DSB directory not found at '{dsb_path}'")
        return
    if not os.path.exists(demographics_path):
        print(f"Error: Demographics file not found at '{demographics_path}'")
        return
        
    print(f"Loading demographics data from: {demographics_path}")
    
    # --- 2. Load and Prepare the Demographics Data ---
    try:
        demo_df = pd.read_csv(demographics_path)
        
        # We only need these columns
        demo_df = demo_df[['PatientID', 'Sex', 'Age']]
        
        # The 'PatientID' is like 'DSB_train_1'. We need to extract 'train' and '1'.
        # We use .str.split() which creates a new DataFrame with the split parts.
        id_parts = demo_df['PatientID'].str.split('_', expand=True)
        
        
        # Assign the extracted parts to new columns
        demo_df['split'] = id_parts[1] # 'train', 'test', or 'val'
        demo_df['pid'] = pd.to_numeric(id_parts[2], errors='coerce') # '1', '10', '100', etc.
        
        # Drop rows where 'pid' could not be converted to a number
        demo_df.dropna(subset=['pid'], inplace=True)
        
        # Convert 'pid' to integer for a clean merge
        demo_df['pid'] = demo_df['pid'].astype(int)
        
        print("Demographics data prepared successfully.")
        print("Sample of prepared data:")
        print(demo_df.head())
        
    except Exception as e:
        print(f"Error processing demographics file: {e}")
        return

    # --- 3. Loop Through Metadata Files and Merge ---
    for split in ['train', 'test', 'val']:
        metadata_filename = f"{split}_metadata_additional.csv"
        metadata_path = os.path.join(dsb_path, metadata_filename)
        
        if not os.path.exists(metadata_path):
            print(f"\nSkipping: Metadata file not found at '{metadata_path}'")
            continue
            
        print(f"\nProcessing: {metadata_filename}")
        
        try:
            # Load the target metadata file
            meta_df = pd.read_csv(metadata_path)
            
            # Filter the demographics data for the current split
            current_demo_df = demo_df[demo_df['split'] == split]
            
            # --- The Core Logic: The Merge ---
            # 'how="left"' ensures we keep all rows from the original metadata file.
            # If a patient in meta_df doesn't have a match in current_demo_df,
            # their 'Sex' and 'Age' will be filled with NaN (Not a Number).
            updated_df = pd.merge(
                meta_df, 
                current_demo_df[['pid', 'Sex', 'Age']], 
                on='pid', 
                how='left'
            )
            
            # Sanity check to ensure we didn't accidentally add/remove rows
            if len(updated_df) != len(meta_df):
                print(f"  [Warning] Row count changed after merge for {metadata_filename}. Check for duplicate PIDs.")
                continue

            # Overwrite the original file with the updated data
            updated_df.to_csv(metadata_path, index=False)
            
            print(f"  Successfully merged and saved. New columns 'Sex' and 'Age' added.")
            print("  New structure:")
            print(updated_df.head())

        except Exception as e:
            print(f"  [Error] Failed to process {metadata_filename}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    BASE_DATA_FOLDER = './data'
    DSB_FOLDER = 'DSB_nifti'
    # IMPORTANT: Make sure this is the correct name of your demographics CSV file
    DEMOGRAPHICS_CSV = 'DSB_only.csv' 
    
    # Run the script
    add_demographics_to_metadata(BASE_DATA_FOLDER, DSB_FOLDER, DEMOGRAPHICS_CSV)
    
    print("\n--- All metadata files updated. ---")