import pandas as pd
import os

# --- Configuration ---
metadata_dir = './data/DSB_nifti'  # <-- CHANGE THIS

# Path to the ES frames CSV file you created in the previous step.
es_csv_path = './data/DSB_nifti/es_frames.csv'

# List of metadata files to process
metadata_files = [
    'train_metadata_additional.csv',
    'val_metadata_additional.csv',
    'test_metadata_additional.csv'
]

# --- Script Execution ---

print("--- Starting CSV Merge Process ---")

# 1. Load the ES frames data
try:
    print(f"Loading ES frame data from: {es_csv_path}")
    es_df = pd.read_csv(es_csv_path)
    print("ES frame data loaded successfully.")
    print(f"Found ES data for {len(es_df)} patients.")
except FileNotFoundError:
    print(f"Error: The file '{es_csv_path}' was not found.")
    print("Please make sure you have run the first script ('find_es_frame.py') successfully.")
    exit() # Exit the script if the essential input file is missing

# 2. Iterate and merge each metadata file
for filename in metadata_files:
    input_path = os.path.join(metadata_dir, filename)

    if not os.path.exists(input_path):
        print(f"\nWarning: Metadata file not found, skipping: {input_path}")
        continue

    print(f"\nProcessing file: {filename}")

    # Load the metadata file
    meta_df = pd.read_csv(input_path)
    print(f"  - Original shape: {meta_df.shape}")
    merged_df = pd.merge(meta_df, es_df, on='pid', how='left')

    # Convert the new column to an integer type, handling potential missing values (NaNs)
    if 'es_frame' in merged_df.columns:
        merged_df['es_frame'] = merged_df['es_frame'].astype('Int64') # 'Int64' (capital I) supports NaNs

    print(f"  - Merged shape:   {merged_df.shape}")

    # 4. Save the new, merged file
    # Create a new filename to avoid overwriting the original
    output_filename = filename.replace('.csv', '_with_es.csv')
    output_path = os.path.join(metadata_dir, output_filename)

    merged_df.to_csv(output_path, index=False)
    print(f"  - Successfully saved merged data to: {output_path}")

print("\n--- All files processed! ---")

# Optional: Display the head of the first new file created to verify the result
first_output_file = os.path.join(metadata_dir, metadata_files[0].replace('.csv', '_with_es.csv'))
if os.path.exists(first_output_file):
    print(f"\n--- Verifying output for '{os.path.basename(first_output_file)}' ---")
    final_df_sample = pd.read_csv(first_output_file)
    print(final_df_sample.head())