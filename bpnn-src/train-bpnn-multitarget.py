import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# --- Configuration ---
DATA_FILE = 'C:/Users/alexg/Downloads/NSF REU Code Repo/data/all_fiber_data_combined.csv'
MODELS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/models'
REPORTS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/reports'
MODEL_NAME = 'bpnn_multitarget_optimized'

# --- Preprocessing Parameters ---
# These values MUST match in the prediction script
FIXED_LENGTH = 200      # Resample each specimen to this many data points
INPUT_SPLIT = 0.2       # Use the first 20% of points as input to predict the remaining 80%

# Feature and target columns
FEATURE_COLS = ['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']
TARGET_COLS = ['Flex Stress (MPa)', 'F Strain (mm/mm)']


def create_sequence_samples():
    """
    Loads data, groups by specimen, and processes each specimen into a single
    input/output sample for sequence-to-sequence prediction.
    Input: The first INPUT_SPLIT portion of the curve.
    Output: The remaining portion of the curve.
    """
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        exit()

    df.sort_values(by=['Fiber_Oz', 'Specimen_ID', 'Crosshead (mm)'], inplace=True)
    specimens = df.groupby(['Fiber_Oz', 'Specimen_ID'])

    X_list, y_list, specimen_keys = [], [], []
    
    split_idx = int(FIXED_LENGTH * INPUT_SPLIT)
    # Ensure there's at least one point in the output
    if split_idx >= FIXED_LENGTH - 1:
        print(f"Error: INPUT_SPLIT ({INPUT_SPLIT}) is too high. Must be less than 1 - (1/{FIXED_LENGTH}).")
        exit()

    for name, group in specimens:
        if len(group) < 2:
            continue

        # Resample the full specimen curve to a fixed length
        resampled_group = pd.DataFrame()
        x_original = group['Crosshead (mm)'].values
        x_resampled = np.linspace(x_original.min(), x_original.max(), FIXED_LENGTH)

        for col in FEATURE_COLS:
            resampled_group[col] = np.interp(x_resampled, x_original, group[col].values)

        # Split into input and output sequences
        input_df = resampled_group.iloc[:split_idx]
        output_df = resampled_group.iloc[split_idx:]

        # Flatten the data to create a single vector for the model
        X_flat = input_df[FEATURE_COLS].values.flatten()
        y_flat = output_df[TARGET_COLS].values.flatten()

        X_list.append(X_flat)
        y_list.append(y_flat)
        specimen_keys.append(name)

    print(f"Processed {len(X_list)} samples from {len(specimens)} specimens.")
    return np.array(X_list), np.array(y_list), specimen_keys


def train_and_evaluate():
    """
    Main pipeline: loads data, finds best hyperparameters via GridSearchCV,
    trains the final model, and evaluates performance.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading and creating sequence samples...")
    X, y, specimen_keys = create_sequence_samples()

    if len(X) == 0:
        print("No data available after processing. Exiting.")
        return

    # Split data based on unique specimens to prevent data leakage
    unique_keys = sorted(list(set(specimen_keys)))
    keys_train, keys_test = train_test_split(unique_keys, test_size=0.2, random_state=42)

    train_indices = [i for i, key in enumerate(specimen_keys) if key in keys_train]
    test_indices = [i for i, key in enumerate(specimen_keys) if key in keys_test]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Model Pipeline with Scaling and Hyperparameter Tuning ---
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('mlp', MLPRegressor(max_iter=1500, random_state=42, early_stopping=True, n_iter_no_change=15, tol=1e-5))
    ])

    # Define a more focused parameter grid for tuning
    param_grid = {
        'mlp__hidden_layer_sizes': [(100, 100), (150, 100, 50)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001],
        'mlp__learning_rate': ['adaptive']
    }

    print("\nStarting hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
    grid_search.fit(X_train, y_train)

    print("\nGridSearchCV finished.")
    print(f"Best R² score from cross-validation: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")

    # --- Save the Best Model ---
    best_model_pipeline = grid_search.best_estimator_
    model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_model.joblib')
    joblib.dump(best_model_pipeline, model_path)
    print(f"\nBest model pipeline saved to {model_path}")

    # --- Evaluation on the Hold-out Test Set ---
    print("\nEvaluating the best model on the unseen test set...")
    y_pred = best_model_pipeline.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    print(f"  Final Test Set R² Score: {final_r2:.4f}")

    # --- Save Evaluation Metrics to File ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(REPORTS_DIR, f'{MODEL_NAME}_evaluation.txt')

    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report generated on: {timestamp}\n")
        f.write(f"Model: {MODEL_NAME}\n\n")
        f.write(f"Preprocessing Parameters:\n")
        f.write(f"  FIXED_LENGTH: {FIXED_LENGTH}\n")
        f.write(f"  INPUT_SPLIT: {INPUT_SPLIT}\n\n")
        f.write(f"Best Parameters from GridSearchCV:\n{grid_search.best_params_}\n\n")
        f.write(f"Best cross-validation R² score: {grid_search.best_score_:.4f}\n")
        f.write(f"Final Test Set R² score: {final_r2:.4f}\n")

    print(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    train_and_evaluate()
    print("\nScript finished.")