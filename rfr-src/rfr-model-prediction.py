import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Updated to load the model saved by the new training script
MODEL_NAME = 'randomforestregressor'  # Corresponds to config['name'].lower() in training script
MODEL_PATH = os.path.join("models", f"{MODEL_NAME}_model.joblib")

def load_model(path):
    """Loads a trained model from a file."""
    try:
        model = joblib.load(path)
        print(f"Model loaded successfully from '{path}'")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{path}'.")
        print("Please run 'train-rfr-model.py' first to train and save the model.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

#PREDICTION

def predict_stress(model, input_data):
    """
    Uses the loaded model to predict Flex Stress on new data.

    Args:
        model: The trained scikit-learn model.
        input_data (pd.DataFrame): DataFrame with features for prediction.
                                   Column names must match training features.

    Returns:
        np.ndarray: The predicted values.
    """
    if model is None:
        return None
    try:
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

#Running script

def main():
    """Main function to run the prediction script."""
    # Load the trained Random Forest model
    rfr_model = load_model(MODEL_PATH)

    if rfr_model:
        # Load dataset to fit empirical relationships
        df = pd.read_csv("data/all_fiber_data_combined.csv")
        df.dropna(subset=["Crosshead (mm)", "Load (N)", "F Strain (mm/mm)"], inplace=True)

        # Fit Polynomial models for smoother relationships
        X_strain = df[["F Strain (mm/mm)"]]
        degree = 3  # Degree of the polynomial
        load_model_fit = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        load_model_fit.fit(X_strain, df["Load (N)"])
        crosshead_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        crosshead_model.fit(X_strain, df["Crosshead (mm)"])

        # Generate strain values to predict over
        strain_values = np.linspace(0, 0.05, 500).reshape(-1, 1)
        load_values = load_model_fit.predict(strain_values)
        crosshead_values = crosshead_model.predict(strain_values)

        # Assemble prediction input
        curve_data_to_predict = pd.DataFrame({
            "Crosshead (mm)": crosshead_values,
            "Load (N)": load_values,
            "F Strain (mm/mm)": strain_values.flatten()
        })

        # Predict stress
        predicted_stresses_curve = predict_stress(rfr_model, curve_data_to_predict)

        if predicted_stresses_curve is not None:
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(strain_values, predicted_stresses_curve, label='Predicted Stress-Strain Curve', color='blue')
            plt.title('Predicted Stress-Strain Curve (Polynomial Inputs)')
            plt.xlabel('F Strain (mm/mm)')
            plt.ylabel('Predicted Flex Stress (MPa)')
            plt.legend()
            plt.grid(True)

            # Save
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            plot_filename = os.path.join(reports_dir, "rfr_predicted_curve.png")
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            plt.show()


if __name__ == "__main__":
    main()