import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- Configuration ---
# This MUST point to the optimized model file
MODEL_NAME = 'bpnn_multitarget_optimized'
MODEL_PATH = os.path.join("models", f"{MODEL_NAME}_model.joblib")
DATA_FILE = 'C:/Users/alexg/Downloads/NSF REU Code Repo/data/all_fiber_data_combined.csv'
REPORTS_DIR = "reports"

# --- Preprocessing Parameters ---
# These values MUST match the training script exactly
FIXED_LENGTH = 200
INPUT_SPLIT = 0.2

# Feature and target columns (must match training script)
FEATURE_COLS = ['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']
TARGET_COLS = ['Flex Stress (MPa)', 'F Strain (mm/mm)']


def load_model_pipeline(model_path):
    """Loads a trained model pipeline (including scaler) from a file."""
    try:
        model_pipeline = joblib.load(model_path)
        print(f"Model pipeline loaded successfully from '{model_path}'")
        return model_pipeline
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the model has been trained and the path is correct.")
        return None


def preprocess_and_predict(model_pipeline, specimen_df):
    """
    Preprocesses a single specimen's data and returns the full actual
    and predicted stress-strain curves.
    """
    if len(specimen_df) < 2:
        return None, None, None, None

    # 1. Resample to the fixed length used in training
    resampled_group = pd.DataFrame()
    x_original = specimen_df['Crosshead (mm)'].values
    x_resampled = np.linspace(x_original.min(), x_original.max(), FIXED_LENGTH)
    for col in FEATURE_COLS:
        resampled_group[col] = np.interp(x_resampled, x_original, specimen_df[col].values)

    # 2. Split into the known input part and the part to be predicted
    split_idx = int(FIXED_LENGTH * INPUT_SPLIT)
    input_df = resampled_group.iloc[:split_idx]

    # 3. Flatten input features for the model
    X_flat = input_df[FEATURE_COLS].values.flatten().reshape(1, -1)

    # 4. Use the pipeline to scale and predict in one step
    # The pipeline handles the scaling before prediction
    predicted_continuation_flat = model_pipeline.predict(X_flat)[0]

    # 5. Reconstruct the full curves for plotting
    # Actual curves
    full_actual_stress = resampled_group['Flex Stress (MPa)'].values
    full_actual_strain = resampled_group['F Strain (mm/mm)'].values
    
    # Predicted curves
    # Reshape the flattened 1D output array back into 2D (points, targets)
    num_predicted_points = FIXED_LENGTH - split_idx
    predicted_continuation = predicted_continuation_flat.reshape(num_predicted_points, len(TARGET_COLS))

    pred_stress_cont = predicted_continuation[:, 0]
    pred_strain_cont = predicted_continuation[:, 1]
    
    # Get the "known" input part of the curve
    input_stress = full_actual_stress[:split_idx]
    input_strain = full_actual_strain[:split_idx]

    # Combine the known input with the predicted continuation
    full_pred_stress = np.concatenate([input_stress, pred_stress_cont])
    full_pred_strain = np.concatenate([input_strain, pred_strain_cont])
    
    return full_actual_strain, full_actual_stress, full_pred_strain, full_pred_stress


def plot_prediction_curve(actual_strain, actual_stress, pred_strain, pred_stress, specimen_key):
    """Generates and saves a plot comparing actual vs. predicted stress-strain curves."""
    split_point_strain = actual_strain[int(FIXED_LENGTH * INPUT_SPLIT) - 1]

    plt.figure(figsize=(12, 7))
    plt.plot(actual_strain, actual_stress, 'b-', label='Actual Curve', linewidth=2.5, alpha=0.8)
    plt.plot(pred_strain, pred_stress, 'r--', label='Predicted Curve', linewidth=2)
    plt.axvline(x=split_point_strain, color='g', linestyle=':', label=f'Input/Prediction Split ({INPUT_SPLIT*100:.0f}%)', linewidth=2)
    
    plt.title(f'{MODEL_NAME.upper()} Prediction vs. Actual for Specimen: {specimen_key}')
    plt.xlabel('F Strain (mm/mm)')
    plt.ylabel('Flex Stress (MPa)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f'{MODEL_NAME}_prediction_specimen_{specimen_key[0]}_{specimen_key[1]}.png'
    plot_path = os.path.join(REPORTS_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved prediction plot to: {plot_path}")


def main():
    """Main execution function to load the model and generate plots for all specimens."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model_pipeline = load_model_pipeline(MODEL_PATH)
    if model_pipeline is None:
        return

    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    df.sort_values(by=['Fiber_Oz', 'Specimen_ID', 'Crosshead (mm)'], inplace=True)
    specimens = df.groupby(['Fiber_Oz', 'Specimen_ID'])

    print(f"\nGenerating predictions for {len(specimens)} specimens...")
    for name, group in specimens:
        actual_strain, actual_stress, pred_strain, pred_stress = preprocess_and_predict(model_pipeline, group)
        if actual_strain is not None:
            plot_prediction_curve(actual_strain, actual_stress, pred_strain, pred_stress, name)

    print("\nPrediction script finished.")


if __name__ == "__main__":
    main()