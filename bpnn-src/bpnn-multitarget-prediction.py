import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_NAME = 'bpnn_multitarget'
MODEL_PATH = os.path.join("models", f"{MODEL_NAME}_model.joblib")
DATA_FILE = 'C:/Users/alexg/Downloads/NSF REU Code Repo/data/all_fiber_data_combined.csv'
REPORTS_DIR = "reports"

FIXED_LENGTH = 100  # Must match the training script's FIXED_LENGTH
INPUT_SPLIT = 0.2   # Must match the training script's INPUT_SPLIT

# Feature and target columns (must match training script)
FEATURE_COLS = ['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']
TARGET_COLS = ['Flex Stress (MPa)', 'F Strain (mm/mm)']

# --- Helper Functions ---

def load_model(path):
    """Loads a trained model from a file."""
    try:
        model = joblib.load(path)
        print(f"Model loaded successfully from '{path}'")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{path}'. Please ensure the model is trained.")
        return None

def preprocess_specimen_data(specimen_df):
    """
    Preprocesses a single specimen's data for prediction.
    Resamples, splits into input/output parts, and flattens the input.
    Returns the flattened input features and the actual full curve for plotting.
    """
    if len(specimen_df) < 2:
        return None, None, None

    # Resample to a fixed length
    resampled_group = pd.DataFrame()
    x_original = specimen_df['Crosshead (mm)'].values
    x_resampled = np.linspace(x_original.min(), x_original.max(), FIXED_LENGTH)

    for col in FEATURE_COLS:
        resampled_group[col] = np.interp(x_resampled, x_original, specimen_df[col].values)

    # Split into input (20%) and target (80%)
    split_idx = int(FIXED_LENGTH * INPUT_SPLIT)
    input_df = resampled_group.iloc[:split_idx]
    output_df = resampled_group.iloc[split_idx:]

    # Flatten input features for model prediction
    X_flat = input_df[FEATURE_COLS].values.flatten()

    # Return the full resampled curve for plotting actuals
    full_actual_stress = resampled_group['Flex Stress (MPa)'].values
    full_actual_strain = resampled_group['F Strain (mm/mm)'].values

    return X_flat, full_actual_stress, full_actual_strain

def plot_prediction(model, X_flat, full_actual_stress, full_actual_strain, specimen_key):
    """
    Generates and saves a plot comparing actual vs. predicted stress-strain curves.
    """
    # Predict the flattened continuation
    y_pred_flat = model.predict(X_flat.reshape(1, -1))[0] # Reshape for single prediction

    num_input_points = int(FIXED_LENGTH * INPUT_SPLIT)
    num_output_points = FIXED_LENGTH - num_input_points

    # Reshape predicted flattened output back to stress and strain
    pred_stress_cont = y_pred_flat[::len(TARGET_COLS)]
    pred_strain_cont = y_pred_flat[1::len(TARGET_COLS)]

    # Extract input part of the curve from full_actual_strain/stress
    input_stress = full_actual_stress[:num_input_points]
    input_strain = full_actual_strain[:num_input_points]

    # Combine input part with predicted continuation for the full predicted curve
    full_pred_strain = np.concatenate([input_strain, pred_strain_cont])
    full_pred_stress = np.concatenate([input_stress, pred_stress_cont])

    plt.figure(figsize=(12, 7))
    plt.plot(full_actual_strain, full_actual_stress, 'b-', label='Actual Curve', linewidth=2)
    plt.plot(full_pred_strain, full_pred_stress, 'r--', label='Predicted Curve', linewidth=2)
    plt.axvline(x=input_strain[-1], color='g', linestyle=':', label='Input/Prediction Split')
    
    plt.title(f'{MODEL_NAME.upper()} Prediction vs. Actual for Specimen: {specimen_key}')
    plt.xlabel('F Strain (mm/mm)')
    plt.ylabel('Flex Stress (MPa)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f'{MODEL_NAME}_prediction_specimen_{specimen_key[0]}_{specimen_key[1]}_curve.png'
    plot_path = os.path.join(REPORTS_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved prediction plot: {plot_path}")

# --- Main Execution ---

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load the trained model
    model = load_model(MODEL_PATH)
    if model is None:
        return

    # Load all data and group by specimen
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    df.sort_values(by=['Fiber_Oz', 'Specimen_ID', 'Crosshead (mm)'], inplace=True)
    specimens = df.groupby(['Fiber_Oz', 'Specimen_ID'])

    print("Generating predictions and plots for each specimen...")
    for name, group in specimens:
        X_flat, full_actual_stress, full_actual_strain = preprocess_specimen_data(group)
        if X_flat is not None:
            plot_prediction(model, X_flat, full_actual_stress, full_actual_strain, name)

    print("\nPrediction script finished.")

if __name__ == "__main__":
    main()