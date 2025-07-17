
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- Configuration ---
MODELS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/models'
REPORTS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/reports'
DATA_FILE = 'C:/Users/alexg/Downloads/NSF REU Code Repo/data/all_fiber_data_combined.csv'
MODEL_NAME = 'rnn_multitarget'

# --- Preprocessing Parameters ---
FIXED_LENGTH = 200
INPUT_SPLIT = 0.2
FEATURE_COLS = ['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']
TARGET_COLS = ['Flex Stress (MPa)', 'F Strain (mm/mm)']


def predict_and_plot_specimen(specimen_name):
    """
    Loads a trained model, makes a prediction on a single specimen, 
    and plots the actual vs. predicted curve.
    """
    # --- Load Model and Scalers ---
    model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_model.keras')
    scaler_X_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_scaler_X.joblib')
    scaler_y_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_scaler_y.joblib')

    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
        print("Error: Model or scalers not found. Please run the training script first.")
        return

    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # --- Load and Prepare Data ---
    df = pd.read_csv(DATA_FILE)
    specimen_data = df[(df['Fiber_Oz'] == specimen_name[0]) & (df['Specimen_ID'] == specimen_name[1])].copy()
    
    if specimen_data.empty:
        print(f"Specimen {specimen_name} not found.")
        return

    # Resample to fixed length
    x_original = specimen_data['Crosshead (mm)'].values
    x_resampled = np.linspace(x_original.min(), x_original.max(), FIXED_LENGTH)
    resampled_group = pd.DataFrame()
    for col in FEATURE_COLS:
        resampled_group[col] = np.interp(x_resampled, x_original, specimen_data[col].values)

    # Prepare input sequence
    split_idx = int(FIXED_LENGTH * INPUT_SPLIT)
    input_df = resampled_group.iloc[:split_idx]
    X_sample = input_df[FEATURE_COLS].values
    X_sample_scaled = scaler_X.transform(X_sample).reshape(1, X_sample.shape[0], X_sample.shape[1])

    # --- Prediction ---
    y_pred_scaled = model.predict(X_sample_scaled)
    y_pred_flat = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = y_pred_flat.reshape(-1, len(TARGET_COLS))

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Actual curve
    plt.plot(specimen_data['F Strain (mm/mm)'], specimen_data['Flex Stress (MPa)'], 'b-', label='Actual Curve')

    # Predicted curve
    predicted_strain = y_pred[:, 1]
    predicted_stress = y_pred[:, 0]
    plt.plot(predicted_strain, predicted_stress, 'r--', label='Predicted Curve')

    plt.title(f'Stress-Strain Curve Prediction for Specimen {specimen_name}')
    plt.xlabel('Strain (mm/mm)')
    plt.ylabel('Stress (MPa)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plot_path = os.path.join(REPORTS_DIR, f'{MODEL_NAME}_prediction_{specimen_name}.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == '__main__':
    # Example: Predict for a specific specimen
    # You can change this to any specimen from your dataset
    specimen_to_predict = ('2-Oz', 1) # (Fiber_Oz, Specimen_ID)
    # Other examples:
    # specimen_to_predict = ('10-Oz', 1)
    # specimen_to_predict = ('4-Oz', 1)
    # specimen_to_predict = ('6-Oz', 1)
    predict_and_plot_specimen(specimen_to_predict)
