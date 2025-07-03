import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib

# --- DATA PREPARATION ---

# Load data from CSV file
try:
    data_path = os.path.join("data", "all_fiber_data_combined.csv")
    data = pd.read_csv(data_path)
    print("File loaded.")
except FileNotFoundError:
    print("File not found.")
    exit()

# drop rows with any missing values
data.dropna(inplace=True)

# Define features (X) and target (y)
features = ["Crosshead (mm)", "Load (N)", "F Strain (mm/mm)"]
target = "Flex Stress (MPa)"

X = data[features]
y = data[target]

# Split the data into training and testing set 80% for training 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("-" * 30)

#MODEL AND HYPERPARAMETER CONFIGURATION 

# A smaller grid for quicker testing. You can expand this for a more thorough search.
param_grid_rfr = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# To add other models, define their parameter grids and add them to this list.
# For example:
# from sklearn.svm import SVR
# param_grid_svr = {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}
#
# model_configs = [
#     {'name': 'RandomForestRegressor', 'estimator': RandomForestRegressor(random_state=42), 'params': param_grid_rfr},
#     {'name': 'SVR', 'estimator': SVR(), 'params': param_grid_svr}
# ]

model_configs = [
    {
        'name': 'RandomForestRegressor',
        'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': param_grid_rfr
    }
]

# --- MODEL TRAINING AND TUNING ---

for config in model_configs:
    model_name = config['name']
    estimator = config['estimator']
    params = config['params']

    print(f"--- Tuning hyperparameters for {model_name} ---")

    # Create a GridSearchCV object
    # cv=5 means 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='r2'
    )

    print(f"Training and tuning {model_name}...")
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    print("Training and tuning complete.")
    print("-" * 30)

    # --- BEST MODEL ---
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {model_name}:")
    print(grid_search.best_params_)
    print(f"Best cross-validation R² score: {grid_search.best_score_:.4f}")
    print("-" * 30)

    # --- SAVING MODEL ---
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    # Save the model with a name that includes the model type
    model_filename = os.path.join(output_dir, f"{model_name.lower()}_model.joblib")
    joblib.dump(best_model, model_filename)
    print(f"Best model for {model_name} saved to {model_filename}")
    print("-" * 30)

    # --- VALIDATION AND EVALUATION ---
    print(f"Evaluating {model_name} on the test set...")
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²) Score: {r2:.4f}")
    print("-" * 30)

    #Feautre Importance
    if hasattr(best_model, 'feature_importances_'):
        print("Feature Importances:")
        importances = pd.Series(best_model.feature_importances_, index=features)
        print(importances.sort_values(ascending=False))
        print("-" * 30)
