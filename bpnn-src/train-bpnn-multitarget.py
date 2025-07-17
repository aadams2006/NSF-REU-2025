import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
DATA_FILE = 'C:/Users/alexg/Downloads/NSF REU Code Repo/data/all_fiber_data_combined.csv'
MODELS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/models'
REPORTS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/reports'
MODEL_NAME = 'bpnn_multitarget'
FIXED_LENGTH = 200  # Resample each specimen to this many data points
INPUT_LENGTH = 20   # Use 20 points as input (10% of 200)

# Feature and target columns
FEATURE_COLS = ['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']
TARGET_COLS = ['Flex Stress (MPa)', 'F Strain (mm/mm)']


def load_and_reshape_data():
    """
    Loads data, groups by specimen, and uses a sliding window to generate
    multiple samples from each specimen.
    """
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        exit()

    df.sort_values(by=['Fiber_Oz', 'Specimen_ID', 'Crosshead (mm)'], inplace=True)
    specimens = df.groupby(['Fiber_Oz', 'Specimen_ID'])

    X_list, y_list, specimen_keys = [], [], []

    for name, group in specimens:
        if len(group) < 2:
            continue

        # Resample to a fixed length
        resampled_group = pd.DataFrame()
        x_original = group['Crosshead (mm)'].values
        x_resampled = np.linspace(x_original.min(), x_original.max(), FIXED_LENGTH)

        for col in FEATURE_COLS:
            resampled_group[col] = np.interp(x_resampled, x_original, group[col].values)

        # Use a sliding window to create more samples
        for i in range(len(resampled_group) - INPUT_LENGTH):
            input_df = resampled_group.iloc[i : i + INPUT_LENGTH]
            output_df = resampled_group.iloc[i + INPUT_LENGTH]

            X_flat = input_df[FEATURE_COLS].values.flatten()
            y_flat = output_df[TARGET_COLS].values.flatten()

            X_list.append(X_flat)
            y_list.append(y_flat)
            specimen_keys.append(name)

    print(f"Processed {len(X_list)} samples from {len(specimens)} specimens.")
    return np.array(X_list), np.array(y_list), specimen_keys


def train_evaluate_and_plot():
    """
    Main pipeline: loads data, trains the BPNN model, evaluates performance,
    and generates prediction plots.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading and reshaping data with sliding window...")
    X, y, specimen_keys = load_and_reshape_data()

    if len(X) == 0:
        print("No data available after processing. Exiting.")
        return

    # --- Corrected Train-Test Split ---
    # Get unique specimen keys
    unique_keys = sorted(list(set(specimen_keys)))
    
    # Split the unique keys into training and testing sets
    keys_train, keys_test = train_test_split(unique_keys, test_size=0.2, random_state=42)

    print("Training specimens:", keys_train)
    print("Testing specimens:", keys_test)

    # Create the training and testing sets for X and y
    train_indices = [i for i, key in enumerate(specimen_keys) if key in keys_train]
    test_indices = [i for i, key in enumerate(specimen_keys) if key in keys_test]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]


    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Data Scaling ---
    print("\nScaling data...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    scaler_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # --- Model Training ---
    print("\nTraining MLPRegressor (BPNN) model...")
    model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', alpha=0.0001, 
                         learning_rate='adaptive', max_iter=2000, random_state=42, tol=1e-4)

    # Fit the model
    model.fit(X_train_scaled, y_train)

    model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # --- Evaluation ---
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    print(f"  R² Score: {r2:.4f}")

    # --- Save Evaluation Metrics to File ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluation_report_path = os.path.join(REPORTS_DIR, f'{MODEL_NAME}_evaluation.txt')

    with open(evaluation_report_path, 'a') as f:
        f.write(f"\n\nEvaluation Report generated on: {timestamp}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
    print(f"Evaluation metrics appended to {evaluation_report_path}")

if __name__ == "__main__":
    train_evaluate_and_plot()
    print("\nScript finished.")