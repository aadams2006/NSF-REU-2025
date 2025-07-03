import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- 1. Data Preparation ---

# Load your data from the CSV file located in the 'data' folder
try:
    data_path = os.path.join("data", "all_fiber_data_combined.csv")
    data = pd.read_csv(data_path)
    print(f"Successfully loaded '{data_path}'.")
except FileNotFoundError:
    print(f"Error: '{data_path}' not found. Make sure you have run aggregated-data.py first.")
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


# --- 2. Model Initialization ---

# Create a K-Nearest Neighbors Regressor model instance
# n_neighbors: Number of neighbors to use. A common starting point is 5.
knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
# n_jobs=-1 uses all available CPU cores for training, which can speed it up.


# --- 3. Model Training ---

print("Training the K-Nearest Neighbors Regressor...")
# Fit the model to the training data
knn.fit(X_train, y_train)
print("Training complete.")
print("-" * 30)


# --- 4. Model Evaluation ---

# Use the trained model to make predictions on the test set
y_pred = knn.predict(X_test)

#Calculate the Mean Squared Error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

#Saving model iteration
output_dir = "models"
model_filename = os.path.join(output_dir, "knn_model.joblib")
joblib.dump(knn, model_filename)
print(f"Model saved to {model_filename}")
print("-" * 30)
