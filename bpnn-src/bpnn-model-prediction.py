import pandas as pd
import joblib
import os

#Load model
MODEL_PATH = os.path.join("models", "bpnn_model.joblib")

def load_model(path):
    """Loads a trained model pipeline from a file."""
    try:
        model = joblib.load(path)
        print(f"Model pipeline loaded successfully from '{path}'")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{path}'.")
        print("Please run 'train-bpnn-model.py' first to train and save the model.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None


#Predicting w/ new data

def predict_stress(model, input_data):
    """
    Uses the loaded model pipeline to predict Flex Stress on new data.

    Args:
        model: The trained scikit-learn pipeline.
        input_data (pd.DataFrame): DataFrame with features for prediction.
                                   Column names must match training features.

    Returns:
        np.ndarray: The predicted values.
    """
    if model is None:
        return None
    try:
        # The pipeline will automatically scale the input_data before prediction
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


#Running script

def main():
    """Main function to run the prediction script."""
    # Load the model pipeline
    bpnn_model_pipeline = load_model(MODEL_PATH)

    if bpnn_model_pipeline:
        # Create new data points as a pandas DataFrame.
        # Column names must EXACTLY match the 'features' used for training.
        new_data_to_predict = pd.DataFrame({
            "Crosshead (mm)": [1.5, 2.0],
            "Load (N)": [180, 250],
            "F Strain (mm/mm)": [0.022, 0.030]
        })

        # Use the loaded model to make predictions
        predicted_stresses = predict_stress(bpnn_model_pipeline, new_data_to_predict)

        if predicted_stresses is not None:
            print("\n" + "-" * 30)
            print("Prediction on New Data:")
            print(f"Input Features:\n{new_data_to_predict.to_string(index=False)}\n")

            for i, prediction in enumerate(predicted_stresses):
                print(f"Prediction for sample {i+1}:")
                print(f"Predicted Flex Stress (MPa): {prediction:.4f}")
            print("-" * 30)


if __name__ == "__main__":
    main()