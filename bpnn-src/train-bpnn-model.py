import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

# --- 1. Data Preparation ---

# Load your data from the CSV file located in the 'data' folder
# This assumes the script is run from the root of the NSF REU Code Repo
try:
    data_path = os.path.join("data", "all_fiber_data_combined.csv")
    data = pd.read_csv(data_path)    
    print("File loaded.")
except FileNotFoundError:
    print("File not found.")
    exit()
# Prepare the data: drop rows with any missing values
data.dropna(inplace=True)

# Define features (X) and target (y)
# We want to predict 'Flex Stress (MPa)' based on other measurements.
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


#Parameter Setup

# Create a pipeline that first scales the data, then applies the BPNN (MLPRegressor)
# Neural Networks are sensitive to feature scaling, so this is a crucial step.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('bpnn', MLPRegressor(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',             # Rectified Linear Unit activation function
        solver='adam',                 # Efficient stochastic gradient-based optimizer
        max_iter=1000,                 # Maximum number of iterations
        random_state=42,
        early_stopping=True,           # Stop training when validation score is not improving
        n_iter_no_change=10            # Number of iterations with no improvement to wait
    ))
])


#Training

print("Training the Backpropagation Neural Network (BPNN)...")
# Fit the pipeline to the training data
# The pipeline will handle scaling the data before feeding it to the MLPRegressor
pipeline.fit(X_train, y_train)
print("Training complete.")
print("-" * 30)


#Evaluation/Validation

# Use the trained pipeline to make predictions on the test set
# The pipeline automatically applies the same scaling to the test data
y_pred = pipeline.predict(X_test)

# Calculate the Mean Squared Error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")
print("-" * 30)

# --- Feature Importance ---
# For models like MLPRegressor, a direct `feature_importances_` attribute is not
# available. We use permutation importance instead, which is a model-agnostic
# method to evaluate how much each feature contributes to the model's accuracy.
print("Feature Importances (Permutation Method):")

result = permutation_importance(
    pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Organize the results into a pandas Series for easy viewing
importances = pd.Series(result.importances_mean, index=features)
print(importances.sort_values(ascending=False))
print("-" * 30)


#Saving Model
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
model_filename = os.path.join(output_dir, "bpnn_model.joblib")
joblib.dump(pipeline, model_filename)
print(f"Model pipeline saved to {model_filename}")
print("-" * 30)