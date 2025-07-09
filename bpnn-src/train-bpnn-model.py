import os
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

# Data Preparation

# Load data assuming the script is run from the root of the repo
try:
    data_path = os.path.join("data", "all_fiber_data_combined.csv")
    data = pd.read_csv(data_path)    
    print("File loaded.")
except FileNotFoundError:
    print("File not found.")
    exit()
# Prepare the data: drop rows with any missing values
data.dropna(inplace=True)

# Define features (X) and target (y) to predict 'Flex Stress (MPa)' based on other measurements
features = ["Crosshead (mm)", "Load (N)", "F Strain (mm/mm)"]
target = "Flex Stress (MPa)"

X = data[features]
y = data[target]

# Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("-" * 30)

# MODEL AND HYPERPARAMETER CONFIGURATION
model_name = 'BPNN (MLPRegressor)'
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('bpnn', MLPRegressor(
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    ))
])

param_grid = {
    'bpnn__hidden_layer_sizes': [(50, 50), (100,), (100, 50)],
    'bpnn__activation': ['relu', 'tanh'],
    'bpnn__solver': ['adam'],
    'bpnn__alpha': [0.0001, 0.001, 0.01],
}

# Create reports directory
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)
report_filename = os.path.join(reports_dir, "bpnn_evaluation.txt")

# Open the report file to write the results
with open(report_filename, 'a') as report_file:
    report_file.write(f"Evaluation Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file.write(f"--- Results for {model_name} generated on {timestamp} ---\n\n")

    # --- Pre-optimization evaluation ---
    print(f"--- Evaluating default {model_name} (pre-optimization) ---")
    default_model_pipeline = pipeline
    default_model_pipeline.fit(X_train, y_train)
    y_pred_default = default_model_pipeline.predict(X_test)
    
    mse_default = mean_squared_error(y_test, y_pred_default)
    r2_default = r2_score(y_test, y_pred_default)

    print("Default Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse_default:.4f}")
    print(f"R-squared (R²) Score: {r2_default:.4f}")
    print("-" * 30)

    report_file.write("Default Model (pre-optimization):\n")
    report_file.write(f"  Mean Squared Error (MSE): {mse_default:.4f}\n")
    report_file.write(f"  R-squared (R²) Score: {r2_default:.4f}\n\n")

    # --- Hyperparameter Tuning ---
    print(f"--- Tuning hyperparameters for {model_name} ---")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        verbose=1,
        n_jobs=-1
    )

    print(f"Training and tuning {model_name}...")
    grid_search.fit(X_train, y_train)
    print("Training and tuning complete.")
    print("-" * 30)

    #Best model
    best_pipeline = grid_search.best_estimator_
    print(f"Best parameters for {model_name}:")
    print(grid_search.best_params_)
    print(f"Best cross-validation R² score: {grid_search.best_score_:.4f}")
    print("-" * 30)

    # Save model iteration
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, "bpnn_model.joblib")
    joblib.dump(best_pipeline, model_filename)
    print(f"Best model for {model_name} saved to {model_filename}")
    print("-" * 30)

    # --- Post-optimization evaluation ---
    print(f"--- Evaluating {model_name} on the test set (post-optimization) ---")
    y_pred = best_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Optimized Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²) Score: {r2:.4f}")
    print("-" * 30)

    report_file.write("Optimized Model (post-GridSearchCV):\n")
    report_file.write(f"  Best Parameters: {grid_search.best_params_}\n")
    report_file.write(f"  Best cross-validation R² score: {grid_search.best_score_:.4f}\n")
    report_file.write(f"  Test Set MSE: {mse:.4f}\n")
    report_file.write(f"  Test Set R² Score: {r2:.4f}\n\n")

    #Feature Importance
    print("Feature Importances (Permutation Method):")
    result = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=features)
    sorted_importances = importances.sort_values(ascending=False)
    print(sorted_importances)
    print("-" * 30)

    report_file.write("Feature Importances (for optimized model):\n")
    report_file.write(f"{sorted_importances.to_string()}\n\n")

print(f"Evaluation report saved to {report_filename}")