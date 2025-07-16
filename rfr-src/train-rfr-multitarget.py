
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import datetime
import joblib

# CONFIGURATION 
DATA_PATH = os.path.join("data", "all_fiber_data_combined.csv")
REPORTS_DIR = "reports"
MODELS_DIR = "models"
VALIDATION_DATA_DIR = os.path.join("data", "validation")
MODEL_NAME = 'RandomForestRegressor_MultiTarget'

# DATA LOADING & PREPARATION
try:
    data = pd.read_csv(DATA_PATH)
    print("File loaded.")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

data.dropna(inplace=True)

# VALIDATION SET CREATION
VALIDATION_FIBER_OZ = '6-Oz'
VALIDATION_SPECIMEN_ID = 3

validation_mask = (data['Fiber_Oz'] == VALIDATION_FIBER_OZ) & (data['Specimen_ID'] == VALIDATION_SPECIMEN_ID)
validation_set = data[validation_mask]
training_data = data[~validation_mask]

if validation_set.empty:
    print("Error: Could not find the specified validation specimen. Please check the IDs.")
    exit()

print(f"Holding out specimen ({VALIDATION_FIBER_OZ}, {VALIDATION_SPECIMEN_ID}) for validation.")
print(f"Training data size: {len(training_data)}")
print(f"Validation data size: {len(validation_set)}")

os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)
validation_set.to_csv(os.path.join(VALIDATION_DATA_DIR, "rfr_validation_set.csv"), index=False)
print(f"Validation set saved to {os.path.join(VALIDATION_DATA_DIR, 'rfr_validation_set.csv')}")

# FEATURE ENGINEERING 
features = ["Crosshead (mm)", "Load (N)", "Flex Stress (MPa)", "F Strain (mm/mm)"]
targets = ["Flex Stress (MPa)", "F Strain (mm/mm)"]

X = training_data[features]
y = training_data[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("-" * 30)

# MODEL TRAINING & HYPERPARAMETER TUNING
param_grid_rfr = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid_rfr,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='r2'
)

print(f"Training and tuning {MODEL_NAME}...")
grid_search.fit(X_train, y_train)
print("Training and tuning complete.")

best_model = grid_search.best_estimator_

# SAVE THE TRAINED MODEL 
os.makedirs(MODELS_DIR, exist_ok=True)
model_filename = os.path.join(MODELS_DIR, f"{MODEL_NAME.lower()}_model.joblib")
joblib.dump(best_model, model_filename)
print(f"Best model saved to {model_filename}")

# EVALUATION & REPORTING 
y_pred = best_model.predict(X_test)
mse_stress = mean_squared_error(y_test["Flex Stress (MPa)"], y_pred[:, 0])
r2_stress = r2_score(y_test["Flex Stress (MPa)"], y_pred[:, 0])
mse_strain = mean_squared_error(y_test["F Strain (mm/mm)"], y_pred[:, 1])
r2_strain = r2_score(y_test["F Strain (mm/mm)"], y_pred[:, 1])

print("\n--- Optimized Model Evaluation on Test Set ---")
print(f"Stress - MSE: {mse_stress:.4f}, R2: {r2_stress:.4f}")
print(f"Strain - MSE: {mse_strain:.4f}, R2: {r2_strain:.4f}")

# Write evaluation report
os.makedirs(REPORTS_DIR, exist_ok=True)
report_filename = os.path.join(REPORTS_DIR, "rfr_multitarget_evaluation.txt")
with open(report_filename, 'w') as report_file:
    report_file.write(f"Evaluation Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_file.write(f"Validation Specimen: {VALIDATION_FIBER_OZ}, ID {VALIDATION_SPECIMEN_ID}\n\n")
    report_file.write("--- Optimized Model (post-GridSearchCV) ---\n")
    report_file.write(f"  Best Parameters: {grid_search.best_params_}\n")
    report_file.write(f"  Best cross-validation R² score: {grid_search.best_score_:.4f}\n")
    report_file.write(f"  Test Set Stress MSE: {mse_stress:.4f}\n")
    report_file.write(f"  Test Set Stress R² Score: {r2_stress:.4f}\n")
    report_file.write(f"  Test Set Strain MSE: {mse_strain:.4f}\n")
    report_file.write(f"  Test Set Strain R² Score: {r2_strain:.4f}\n\n")

    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=features)
        sorted_importances = importances.sort_values(ascending=False)
        report_file.write("Feature Importances:\n")
        report_file.write(f"{sorted_importances.to_string()}\n")

print(f"\nEvaluation report saved to {report_filename}")

# Save test data for further analysis if needed
X_test.to_csv(os.path.join("data", "rfr_X_test.csv"), index=False)
y_test.to_csv(os.path.join("data", "rfr_y_test.csv"), index=False)
print("Test data (X_test, y_test) saved for further analysis.")
print("-" * 30)
print("Script finished.")
