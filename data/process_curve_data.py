
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'all_fiber_data_combined.csv')
output_dir = base_dir

# Load the data
df = pd.read_csv(data_path)

# Group by specimen
specimens = [group for _, group in df.groupby(['Fiber_Oz', 'Specimen_ID'])]

# Find max length for padding
max_input_len = 0
max_target_len = 0
for specimen in specimens:
    n_rows = len(specimen)
    # Ensure there's at least one row for input and one for target
    if n_rows < 2:
        continue
    split_idx = int(n_rows * 0.2)
    if split_idx == 0:
        split_idx = 1
    
    input_len = split_idx
    target_len = n_rows - input_len
    
    if input_len > max_input_len:
        max_input_len = input_len
    if target_len > max_target_len:
        max_target_len = target_len

# Process data for each specimen
X_list = []
y_list = []

for specimen in specimens:
    n_rows = len(specimen)
    if n_rows < 2:
        continue
        
    split_idx = int(n_rows * 0.2)
    if split_idx == 0:
        split_idx = 1

    input_df = specimen.iloc[:split_idx]
    target_df = specimen.iloc[split_idx:]

    # Input features
    input_features = input_df[['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']].values.flatten()
    
    # Target features
    target_features = target_df[['Flex Stress (MPa)', 'F Strain (mm/mm)']].values.flatten()

    # Pad features
    padded_input = np.pad(input_features, (0, max_input_len * 4 - len(input_features)), 'constant')
    padded_target = np.pad(target_features, (0, max_target_len * 2 - len(target_features)), 'constant')
    
    X_list.append(padded_input)
    y_list.append(padded_target)

X = np.array(X_list)
y = np.array(y_list)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the data as CSV
pd.DataFrame(X_train).to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(output_dir, 'y_train_processed.csv'), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(output_dir, 'y_test_processed.csv'), index=False)

print("Data processing complete.")
print(f"Padded input length (number of timesteps): {max_input_len}")
print(f"Padded target length (number of timesteps): {max_target_len}")
print(f"Number of input features (timesteps * 4): {max_input_len * 4}")
print(f"Number of target features (timesteps * 2): {max_target_len * 2}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
