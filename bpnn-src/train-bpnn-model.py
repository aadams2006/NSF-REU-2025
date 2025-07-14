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

# CONFIGURATION
DATA_PATH = os.path.join("data", "all_fiber_data_combined.csv")
REPORTS_DIR = "reports"
MODELS_DIR = "models"
VALIDATION_DATA_DIR = os.path.join("data", "validation")
MODEL_NAME = 'BPNN (MLPRegressor)'

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
validation_set.to_csv(os.path.join(VALIDATION_DATA_DIR, "bpnn_validation_set.csv"), index=False)
print(f"Validation set saved to {os.path.join(VALIDATION_DATA_DIR, 'bpnn_validation_set.csv')}")

# FEATURE ENGINEERING
features = ["Crosshead (mm)", "Load (N)", "F Strain (mm/mm)"]
target = "Flex Stress (MPa)"

X = training_data[features]
y = training_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("-" * 30)

# MODEL TRAINING & HYPERPARAMETER TUNING
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

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

print(f"Training and tuning {MODEL_NAME}...")
grid_search.fit(X_train, y_train)
print("Training and tuning complete.")

best_pipeline = grid_search.best_estimator_

# SAVE THE TRAINED MODEL
os.makedirs(MODELS_DIR, exist_ok=True)
model_filename = os.path.join(MODELS_DIR, "bpnn_model.joblib")
joblib.dump(best_pipeline, model_filename)
print(f"Best model saved to {model_filename}")

# EVALUATION & REPORTING
y_pred = best_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Optimized Model Evaluation on Test Set ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

os.makedirs(REPORTS_DIR, exist_ok=True)
report_filename = os.path.join(REPORTS_DIR, "bpnn_evaluation.txt")
with open(report_filename, 'w') as report_file:
    report_file.write(f"Evaluation Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_file.write(f"Validation Specimen: {VALIDATION_FIBER_OZ}, ID {VALIDATION_SPECIMEN_ID}\n\n")
    report_file.write("--- Optimized Model (post-GridSearchCV) ---\n")
    report_file.write(f"  Best Parameters: {grid_search.best_params_}\n")
    report_file.write(f"  Best cross-validation R² score: {grid_search.best_score_:.4f}\n")
    report_file.write(f"  Test Set MSE: {mse:.4f}\n")
    report_file.write(f"  Test Set R² Score: {r2:.4f}\n\n")

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

print(f"\nEvaluation report saved to {report_filename}")
