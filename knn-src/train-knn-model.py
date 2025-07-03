import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#Data Preparation

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


#Hyperparameter Tuning with Grid Search

# Create a K-Nearest Neighbors Regressor model instance
knn = KNeighborsRegressor(n_jobs=-1)

# Define the parameter grid to search
# We'll search for the best number of neighbors, weight function, and distance metric.
param_grid = {
    'n_neighbors': range(1, 31),  # Test k from 1 to 30
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Create a GridSearchCV object
# cv=5 means 5-fold cross-validation.
# scoring='r2' will optimize for the best R-squared score.
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=1  # To see the progress
)


#Model Training

print("Running Grid Search to find the best hyperparameters...")
# Fit the grid search to the training data
grid_search.fit(X_train, y_train)
print("Grid Search complete.")
print("-" * 30)

# Get the best model from the grid search
best_knn = grid_search.best_estimator_

# Print the best parameters found
print("Best Hyperparameters Found:")
print(grid_search.best_params_)
print("-" * 30)


#Model Evaluation

#Use the best model to make predictions on the test set
y_pred = best_knn.predict(X_test)

#Calculate the Mean Squared Error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluation of the Best Model:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

#Saving the best model
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
model_filename = os.path.join(output_dir, "knn_model.joblib")
joblib.dump(best_knn, model_filename)
print(f"\nBest model saved to {model_filename}")
print("-" * 30)
