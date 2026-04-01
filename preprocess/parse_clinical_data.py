import os
import csv
import re
# import math # Not strictly needed as round() is built-in
try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom library not found. Processing for 014_SecondAnnualDataScienceBowl will be skipped.")
    print("Please install it using: pip install pydicom")

# --- Configuration ---
# ACDC_DIR = '../data/ACDC/'
DSB_DIR = './data/DSB/' # New Dataset Directory
OUTPUT_CSV = './data/DSB_nifti/DSB_only.csv'
NA_VALUE = 'NA' # Value to use for missing data

# --- Helper Functions ---

def clean_na_string(value_str, na_marker="N/A", true_na_value=NA_VALUE):
    """Converts a specific NA marker string or empty string to the standard NA_VALUE."""
    if value_str is None:
        return true_na_value
    stripped_value = value_str.strip()
    if stripped_value == na_marker or not stripped_value:
        return true_na_value
    return stripped_value

def calculate_bmi(height_cm_val, weight_kg_val):
    """Calculates BMI given height in cm and weight in kg (as float or None)."""
    if height_cm_val is None or weight_kg_val is None:
        return NA_VALUE
    try:
        # Values are already floats or None by the time they reach here
        height_m = height_cm_val / 100.0
        if height_m > 0 and weight_kg_val > 0:
            bmi = weight_kg_val / (height_m ** 2)
            return round(bmi, 2)
        else:
            return NA_VALUE
    except (ValueError, TypeError, ZeroDivisionError): # Added ZeroDivisionError
        return NA_VALUE

def determine_overweight(bmi):
    """Determines overweight status based on BMI."""
    if bmi == NA_VALUE:
        return NA_VALUE
    try:
        if float(bmi) >= 25.0:
            return 'Y'
        else:
            return 'N'
    except (ValueError, TypeError):
        return NA_VALUE

def parse_acdc_info(file_path):
    """Parses the Info.cfg file from the ACDC dataset."""
    info = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
    except FileNotFoundError:
        # print(f"Warning: Info.cfg not found at {file_path}") # Can be verbose
        return None
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}")
        return None
    return info

def parse_dsb_dicom_info(patient_dir_path):
    """
    Parses DICOM files in one of the acquisition subfolders for a DSB patient
    to get Age and Sex.
    """
    if not PYDICOM_AVAILABLE:
        return {'Sex': NA_VALUE, 'Age': NA_VALUE}

    sex = NA_VALUE
    age = NA_VALUE
    dicom_found_for_patient = False

    if not os.path.isdir(patient_dir_path):
        print(f"Warning: DSB Patient directory not found: {patient_dir_path}")
        return {'Sex': sex, 'Age': age}

    for acq_folder_name in os.listdir(patient_dir_path):
        acq_folder_path = os.path.join(patient_dir_path, acq_folder_name)
        if os.path.isdir(acq_folder_path):
            for filename in os.listdir(acq_folder_path):
                if filename.lower().endswith('.dcm'):
                    dicom_file_path = os.path.join(acq_folder_path, filename)
                    try:
                        ds = pydicom.dcmread(dicom_file_path)
                        # PatientSex (0010,0040)
                        if 'PatientSex' in ds and ds.PatientSex:
                            sex_val = ds.PatientSex.upper()
                            if sex_val in ['M', 'F', 'O']: # O for Other
                                sex = sex_val
                        # PatientAge (0010,1010) - e.g., "065Y", "012M"
                        if 'PatientAge' in ds and ds.PatientAge:
                            age_str = str(ds.PatientAge)
                            age_match = re.search(r'(\d+)', age_str) # Extract numbers
                            if age_match:
                                age = age_match.group(1)
                        dicom_found_for_patient = True
                        break # Found one DICOM, that's enough for Age/Sex
                    except InvalidDicomError:
                        print(f"Warning: Invalid DICOM file skipped: {dicom_file_path}")
                    except AttributeError as e:
                        print(f"Warning: Missing DICOM tag in {dicom_file_path}: {e}")
                    except Exception as e:
                        print(f"Warning: Error reading DICOM {dicom_file_path}: {e}")
            if dicom_found_for_patient:
                break # Stop searching acquisition folders for this patient
    
    if not dicom_found_for_patient:
        print(f"Warning: No usable DICOM file found in {patient_dir_path} or its subfolders to extract Age/Sex.")

    return {'Sex': sex, 'Age': age}

# --- Main Processing Logic ---

all_patient_data = []
processed_ids = set()

# 1. Process ACDC Dataset
# print(f"Processing {ACDC_DIR}...")
# if os.path.isdir(ACDC_DIR):
#     for subset in ['training', 'testing']:
#         subset_dir = os.path.join(ACDC_DIR, subset)
#         if os.path.isdir(subset_dir):
#             for patient_folder in os.listdir(subset_dir):
#                 if patient_folder.startswith('patient'):
#                     patient_dir = os.path.join(subset_dir, patient_folder)
#                     if os.path.isdir(patient_dir):
#                         patient_id = f"ACDC_{patient_folder}"
                        
#                         if patient_id in processed_ids:
#                             print(f"Warning: Duplicate PatientID found {patient_id} from ACDC, skipping.")
#                             continue

#                         info_cfg_path = os.path.join(patient_dir, 'Info.cfg')
#                         acdc_info = parse_acdc_info(info_cfg_path)

#                         if acdc_info:
#                             height_str = acdc_info.get('Height', NA_VALUE)
#                             weight_str = acdc_info.get('Weight', NA_VALUE)
#                             pathology = acdc_info.get('Group', NA_VALUE)

#                             height_f = None
#                             if height_str != NA_VALUE: # Check if it's a valid number string
#                                 try:
#                                     height_f = float(height_str)
#                                 except ValueError:
#                                     height_str = NA_VALUE # Revert to NA if not convertible
                            
#                             weight_f = None
#                             if weight_str != NA_VALUE:
#                                 try:
#                                     weight_f = float(weight_str)
#                                 except ValueError:
#                                     weight_str = NA_VALUE # Revert to NA

#                             bmi = calculate_bmi(height_f, weight_f)
#                             overweight = determine_overweight(bmi)

#                             patient_data = {
#                                 'PatientID': patient_id,
#                                 'Dataset': '001_ACDC',
#                                 'Sex': NA_VALUE,
#                                 'Age': NA_VALUE,
#                                 'Height': height_str, # Store original string or NA_VALUE
#                                 'Weight': weight_str, # Store original string or NA_VALUE
#                                 'BMI': bmi,
#                                 'Overweight': overweight,
#                                 'Pathology': pathology,
#                                 'Hypertension': NA_VALUE
#                             }
#                             all_patient_data.append(patient_data)
#                             processed_ids.add(patient_id)
#                         # else:
#                         #      print(f"Could not parse info for {patient_id}") # Can be verbose
# else:
#     print(f"Warning: Directory {ACDC_DIR} not found.")

# 2. Process DSB Dataset
print(f"\nProcessing {DSB_DIR}...")
if not PYDICOM_AVAILABLE:
    print(f"Skipping {DSB_DIR} because pydicom library is not available.")
elif os.path.isdir(DSB_DIR):
    subsets_map = {
        'train': range(1, 501),
        'validate': range(501, 701),
        'test': range(701, 1141) # up to 1140 inclusive
    }
    for subset_name, patient_range in subsets_map.items():
        subset_path = os.path.join(DSB_DIR, subset_name)
        if os.path.isdir(subset_path):
            print(f"Processing DSB subset: {subset_name}")
            # Patient folders are just numbers
            for patient_num_str in os.listdir(subset_path):
                try:
                    patient_num = int(patient_num_str)
                    # Validate if patient_num is expected for this subset (optional, but good for sanity)
                    # if patient_num not in patient_range:
                    #     print(f"Warning: Patient folder {patient_num_str} in {subset_name} seems out of expected range. Processing anyway.")
                    #     pass # Or skip if strict

                    patient_folder_path = os.path.join(subset_path, patient_num_str,"study")
                    if os.path.isdir(patient_folder_path):
                        patient_id = f"DSB_{subset_name}_{patient_num_str}"
                        if patient_id in processed_ids:
                            print(f"Warning: Duplicate PatientID found {patient_id}, skipping.")
                            continue

                        dicom_info = parse_dsb_dicom_info(patient_folder_path)

                        all_patient_data.append({
                            'PatientID': patient_id,
                            'Dataset': '014_DSB', # Shortened name for CSV
                            'Sex': dicom_info.get('Sex', NA_VALUE),
                            'Age': dicom_info.get('Age', NA_VALUE),
                            'Height': NA_VALUE,
                            'Weight': NA_VALUE,
                            'BMI': NA_VALUE,
                            'Overweight': NA_VALUE, # Cannot determine without H/W
                            'Pathology': NA_VALUE, # Not directly available from these DICOMs
                            'Hypertension': NA_VALUE
                        })
                        processed_ids.add(patient_id)
                except ValueError:
                    print(f"Warning: Non-numeric folder name '{patient_num_str}' found in {subset_path}, skipping.")
                except Exception as e:
                    print(f"Error processing patient {patient_num_str} in DSB {subset_name}: {e}")
        else:
            print(f"Warning: DSB subset directory not found: {subset_path}")
else:
    print(f"Warning: Directory {DSB_DIR} not found.")

# 3. Write Output CSV
print(f"\nWriting data to {OUTPUT_CSV}...")
if all_patient_data:
    fieldnames = ['PatientID', 'Dataset', 'Sex', 'Age', 'Height', 'Weight', 'BMI', 'Overweight', 'Pathology', 'Hypertension']
    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_patient_data)
        print(f"Successfully wrote {len(all_patient_data)} records to {OUTPUT_CSV}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")
else:
    print("No patient data found to write.")

print("\nProcessing complete.")