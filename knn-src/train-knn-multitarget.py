import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import os

# --- Configuration ---
DATA_FILE = 'C:/Users/alexg/Downloads/NSF REU Code Repo/data/all_fiber_data_combined.csv'
MODELS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/models'
REPORTS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/reports'
MODEL_NAME = 'knn_multitarget'
FIXED_LENGTH = 100  # Resample each specimen to this many data points
INPUT_SPLIT = 0.2   # First 20% of points for input
N_PLOT_SAMPLES = 3  # Number of specimen plots to generate

# Feature and target columns
FEATURE_COLS = ['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']
TARGET_COLS = ['Flex Stress (MPa)', 'F Strain (mm/mm)']


def load_and_reshape_data():
    """
    Loads data, groups by specimen, and reshapes it according to the
    20% input / 80% target split to prevent data leakage.
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
        if len(group) < 2:  # Need at least 2 points to interpolate
            continue

        # Resample to a fixed length
        resampled_group = pd.DataFrame()
        x_original = group['Crosshead (mm)'].values
        x_resampled = np.linspace(x_original.min(), x_original.max(), FIXED_LENGTH)

        for col in FEATURE_COLS:
            resampled_group[col] = np.interp(x_resampled, x_original, group[col].values)

        # Split into input (20%) and target (80%)
        split_idx = int(FIXED_LENGTH * INPUT_SPLIT)
        input_df = resampled_group.iloc[:split_idx]
        output_df = resampled_group.iloc[split_idx:]

        # Flatten sequences for scikit-learn compatibility
        X_flat = input_df[FEATURE_COLS].values.flatten()
        y_flat = output_df[TARGET_COLS].values.flatten()

        X_list.append(X_flat)
        y_list.append(y_flat)
        specimen_keys.append(name)

    print(f"Processed {len(X_list)} specimens.")
    return np.array(X_list), np.array(y_list), specimen_keys


def train_evaluate_and_plot():
    """
    Main pipeline: loads data, trains the KNN model, evaluates performance,
    and generates prediction plots.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading and reshaping data to prevent leakage...")
    X, y, specimen_keys = load_and_reshape_data()

    if len(X) == 0:
        print("No data available after processing. Exiting.")
        return

    # Split specimens into training and testing sets
    X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(
        X, y, specimen_keys, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Model Training ---
    from sklearn.model_selection import GridSearchCV

    print("\nTraining KNeighborsRegressor model with GridSearchCV...")
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }

    # Initialize KNeighborsRegressor
    knn = KNeighborsRegressor(n_jobs=-1)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    model = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best R2 score found: {grid_search.best_score_:.4f}")

    model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # --- Evaluation ---
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    # Reshape for metric calculation
    num_output_points = FIXED_LENGTH - int(FIXED_LENGTH * INPUT_SPLIT)
    y_test_reshaped = y_test.reshape(-1, num_output_points, len(TARGET_COLS))
    y_pred_reshaped = y_pred.reshape(-1, num_output_points, len(TARGET_COLS))

    r2_stress = r2_score(y_test_reshaped[:, :, 0], y_pred_reshaped[:, :, 0])
    r2_strain = r2_score(y_test_reshaped[:, :, 1], y_pred_reshaped[:, :, 1])

    print(f"  R² Score (Stress): {r2_stress:.4f}")
    print(f"  R² Score (Strain): {r2_strain:.4f}")

    # --- Save Evaluation Metrics to File ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluation_report_path = os.path.join(REPORTS_DIR, f'{MODEL_NAME}_evaluation.txt')

    with open(evaluation_report_path, 'a') as f:
        f.write(f"\n\nEvaluation Report generated on: {timestamp}\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best Cross-Validation R2 Score: {grid_search.best_score_:.4f}\n")
        f.write(f"Test Set R2 Score (Stress): {r2_stress:.4f}\n")
        f.write(f"Test Set R2 Score (Strain): {r2_strain:.4f}\n")
    print(f"Evaluation metrics appended to {evaluation_report_path}")

    

if __name__ == "__main__":
    train_evaluate_and_plot()
    print("\nScript finished.")