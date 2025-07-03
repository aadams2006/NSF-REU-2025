import pandas as pd
import glob
import os
import re

# --- Find all Excel files in the directory ---
excel_files = glob.glob(os.path.join("data", "*.xlsx"))
if not excel_files:
    print("Error: No '.xlsx' files found in the 'data' directory.")
    exit()

print(f"Found {len(excel_files)} Excel file(s) to process: {excel_files}")

# --- PARAMETERS ---
# Data starts at Row 11 (index 10).
# Columns: E-H (starts at index 4), I-L (starts at index 8), and so on.
# The step between the start of each specimen block is 4 columns.
starting_cols = [4, 8, 12, 16, 20]
column_labels = ["Crosshead (mm)", "Load (N)", "Flex Stress (MPa)", "F Strain (mm/mm)"]
DATA_START_ROW = 10  # Row 11 in Excel is index 10

# Initialize list to collect all dataframes from all files
all_data = []

# --- DATA EXTRACTION ---
# Loop through each found Excel file
for excel_file in excel_files:
    print(f"\n--- Processing file: '{excel_file}' ---")
    try:
        xls = pd.ExcelFile(excel_file)
    except Exception as e:
        print(f"Error reading '{excel_file}': {e}")
        continue  # Skip to the next file

    # Derive Fiber_Oz from filename
    filename_base = os.path.splitext(excel_file)[0]
    # Find the part of the filename like "2-QZ" and convert to "2-Oz"
    match = re.search(r"(\d+)-QZ", filename_base, re.IGNORECASE)
    if match:
        fiber_oz_val = f"{match.group(1)}-Oz"
    else:
        fiber_oz_val = filename_base  # Fallback if pattern is not found

    # Loop through each sheet that starts with 'S' (e.g., S1, S2, ...)
    for sheet_name in xls.sheet_names:
        if not sheet_name.startswith("S"):
            continue

        print(f"  Processing sheet: '{sheet_name}'...")
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=None)  # Read without header

        for i, start_col in enumerate(starting_cols):
            specimen_num = i + 1
            try:
                # Define the exact columns for the current specimen block
                end_col = start_col + 4

                # Select the data block using iloc
                df_block = df_sheet.iloc[DATA_START_ROW:, start_col:end_col].copy()

                # Check if the block is empty before proceeding
                if df_block.dropna(how='all').empty:
                    print(f"    - Skipping empty block for Specimen {specimen_num} in sheet '{sheet_name}'.")
                    continue

                # Assign column names and identifiers
                df_block.columns = column_labels
                df_block["Specimen_ID"] = specimen_num
                df_block["Fiber_Oz"] = fiber_oz_val
                all_data.append(df_block)
                print(f"    - Successfully processed Specimen {specimen_num}.")

            except IndexError:
                print(f"    - Warning: Columns for Specimen {specimen_num} not found in sheet '{sheet_name}'.")
            except Exception as e:
                print(f"    - Error processing Specimen {specimen_num} in sheet '{sheet_name}': {e}")


# --- COMBINE, CLEAN, AND SAVE ---
if all_data:
    # Combine all collected dataframes
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Remove any rows where all values are missing
    df_final.dropna(how="all", inplace=True)
    
    # Reorder columns for clarity
    df_final = df_final[["Fiber_Oz", "Specimen_ID"] + column_labels]
    
    # Convert data columns to numeric, coercing errors
    for col in column_labels:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    # Drop rows that may have become empty after numeric conversion
    df_final.dropna(subset=column_labels, how='all', inplace=True)

    # Define the output path to save the CSV inside the 'data' folder
    output_filename = "all_fiber_data_combined.csv"
    output_path = os.path.join("data", output_filename)
    # Save the final combined data to a CSV file
    df_final.to_csv(output_path, index=False)
    print(f"\nCombined CSV saved as '{output_path}'")
else:
    print("\nNo data was extracted. Please check the sheet names and data layout in your Excel file(s).")