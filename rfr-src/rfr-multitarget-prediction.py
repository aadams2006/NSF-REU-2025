import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# CONFIGURATION
MODEL_NAME = 'randomforestregressor_multitarget'
MODEL_PATH = os.path.join("models", f"{MODEL_NAME}_model.joblib")
VALIDATION_SET_PATH = os.path.join("data", "validation", "rfr_validation_set.csv")
REPORTS_DIR = "reports"

# FUNCTION DEFINITIONS

def load_model(path):
    """Loads a trained model from a file."""
    try:
        model = joblib.load(path)
        print(f"Model loaded successfully from '{path}'")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{path}'. Run training first.")
        return None

def load_validation_data(path):
    """Loads the validation dataset from a CSV file."""
    try:
        data = pd.read_csv(path)
        print(f"Validation data loaded successfully from '{path}'.")
        return data
    except FileNotFoundError:
        print(f"Error: Validation data not found at '{path}'. Run training first.")
        return None

def predict_stress_strain(model, input_data):
    """Predicts Flex Stress and Strain using the loaded model."""
    if model is None:
        return None, None
    features = ["Crosshead (mm)", "Load (N)", "Flex Stress (MPa)", "F Strain (mm/mm)"]
    predictions = model.predict(input_data[features])
    return predictions[:, 0], predictions[:, 1]

# MAIN EXECUTION

def main():
    """Main function to run the prediction and plotting script."""
    # 1. Load Model and Validation Data
    rfr_model = load_model(MODEL_PATH)
    if rfr_model is None:
        return

    validation_data = load_validation_data(VALIDATION_SET_PATH)
    if validation_data is None:
        return

    # 2. Make Predictions on the Validation Set
    predicted_stresses, predicted_strains = predict_stress_strain(rfr_model, validation_data)
    if predicted_stresses is None or predicted_strains is None:
        return

    # 3. Prepare Data for Plotting
    plot_data = validation_data.copy()
    plot_data['Predicted_Stress'] = predicted_stresses
    plot_data['Predicted_Strain'] = predicted_strains
    
    # Sort by strain to ensure a continuous line plot
    plot_data.sort_values(by='F Strain (mm/mm)', inplace=True)

    # Get specimen info for the title
    specimen_info = f"{plot_data['Fiber_Oz'].iloc[0]}, Specimen {plot_data['Specimen_ID'].iloc[0]}"

    # 4. Generate and Save the Plot
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['F Strain (mm/mm)'], plot_data['Flex Stress (MPa)'], 'o-', label='Actual Stress-Strain Curve', color='gray', alpha=0.7)
    plt.plot(plot_data['Predicted_Strain'], plot_data['Predicted_Stress'], 'x--', label='Predicted Stress-Strain Curve', color='blue')
    
    plt.title(f'Hold-Out Validation Prediction for: {specimen_info}')
    plt.xlabel('F Strain (mm/mm)')
    plt.ylabel('Flex Stress (MPa)')
    plt.legend()
    plt.grid(True)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plot_filename = os.path.join(REPORTS_DIR, "rfr_multitarget_hold_out_validation_curve.png")
    plt.savefig(plot_filename)
    print(f"Plot for validation specimen saved to {plot_filename}")

    # 5. Create and Save a table with the results
    results_df = pd.DataFrame({
        'Actual Stress': plot_data['Flex Stress (MPa)'],
        'Predicted Stress': plot_data['Predicted_Stress'],
        'Actual Strain': plot_data['F Strain (mm/mm)'],
        'Predicted Strain': plot_data['Predicted_Strain']
    })
    results_filename = os.path.join(REPORTS_DIR, "rfr_multitarget_actual_vs_predicted.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"Results table saved to {results_filename}")

    plt.show()

if __name__ == "__main__":
    main()
