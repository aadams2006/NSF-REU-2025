

"""
This script trains a Recurrent Neural Network (RNN) model, specifically using LSTM layers,
to predict multiple target variables from sequential fiber data. It includes data
preprocessing, model definition, training, evaluation, and saving of the trained
model and scalers.

The script performs the following steps:
1.  **Configuration**: Sets up paths for data, models, and reports, and defines
    model-specific parameters.
2.  **Data Loading and Preprocessing**: Reads the combined fiber data, groups it
    by specimen, and resamples each specimen's data to a fixed length. It then
    splits each specimen's data into input (features) and output (targets) sequences.
3.  **Data Scaling**: Applies Min-Max scaling to both input features and output
    targets to normalize the data for optimal model training.
4.  **Model Definition**: Constructs an LSTM-based RNN model using TensorFlow/Keras.
    The model is designed for sequence-to-sequence prediction, taking a sequence
    of features and outputting a sequence of target values.
5.  **Model Training**: Compiles and trains the RNN model on the prepared and
    scaled training data.
6.  **Model Saving**: Saves the trained RNN model and the fitted scalers (for
    features and targets) to disk for later use in prediction.
7.  **Model Evaluation**: Evaluates the trained model's performance on a held-out
    test set using the R-squared metric.
8.  **Reporting**: Generates and saves an evaluation report with key metrics and
    preprocessing parameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping
import joblib
import os
from datetime import datetime

# --- Configuration ---
# Define file paths and model parameters for consistency and easy modification.
DATA_FILE = 'C:/Users/alexg/Downloads/NSF REU Code Repo/data/all_fiber_data_combined.csv'
MODELS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/models'
REPORTS_DIR = 'C:/Users/alexg/Downloads/NSF REU Code Repo/reports'
MODEL_NAME = 'rnn_multitarget'

# --- Preprocessing Parameters ---
# FIXED_LENGTH: The desired length for each resampled specimen sequence.
# INPUT_SPLIT: The proportion of the fixed-length sequence to be used as input features.
FIXED_LENGTH = 200
INPUT_SPLIT = 0.5 # Proportion of the fixed-length sequence to be used as input features. A higher value means more input data and a shorter prediction sequence.

# Define the columns used as input features and target outputs for the model.
FEATURE_COLS = ['Crosshead (mm)', 'Load (N)', 'Flex Stress (MPa)', 'F Strain (mm/mm)']
TARGET_COLS = ['Flex Stress (MPa)', 'F Strain (mm/mm)']


def create_sequence_samples():
    """
    Loads raw fiber data, groups it by individual specimens, and processes each
    specimen's time-series data into fixed-length input (X) and output (y) sequences.
    This function handles resampling and splitting of data for sequence-to-sequence
    model training.

    Returns:
        tuple: A tuple containing:
            - X (np.array): A 3D numpy array of input sequences,
                            shape (num_samples, input_sequence_length, num_features).
            - y (np.array): A 2D numpy array of flattened target sequences,
                            shape (num_samples, output_sequence_length * num_targets).
            - specimen_keys (list): A list of tuples, where each tuple identifies
                                    a unique specimen (e.g., ('Fiber_Oz', Specimen_ID)).
    """
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        exit()

    # Sort data to ensure proper sequence order within each specimen.
    df.sort_values(by=['Fiber_Oz', 'Specimen_ID', 'Crosshead (mm)'], inplace=True)
    # Group data by unique specimens.
    specimens = df.groupby(['Fiber_Oz', 'Specimen_ID'])

    X_list, y_list, specimen_keys = [], [], []
    
    # Calculate the split point for input and output sequences.
    split_idx = int(FIXED_LENGTH * INPUT_SPLIT)
    if split_idx >= FIXED_LENGTH - 1:
        print(f"Error: INPUT_SPLIT ({INPUT_SPLIT}) is too high. It must leave at least one point for the output sequence.")
        exit()

    for name, group in specimens:
        # Skip specimens with insufficient data points.
        if len(group) < 2:
            continue

        # Resample each feature to the FIXED_LENGTH.
        resampled_group = pd.DataFrame()
        x_original = group['Crosshead (mm)'].values
        # Create a linearly spaced array for resampling.
        x_resampled = np.linspace(x_original.min(), x_original.max(), FIXED_LENGTH)

        for col in FEATURE_COLS:
            # Interpolate values to the new fixed length.
            resampled_group[col] = np.interp(x_resampled, x_original, group[col].values)

        # Split the resampled data into input (X) and output (y) based on INPUT_SPLIT.
        input_df = resampled_group.iloc[:split_idx]
        output_df = resampled_group.iloc[split_idx:]

        # Append processed sequences to lists.
        X_list.append(input_df[FEATURE_COLS].values)
        # Flatten the output targets for the Dense layer in the RNN.
        y_list.append(output_df[TARGET_COLS].values.flatten())
        specimen_keys.append(name)

    print(f"Processed {len(X_list)} samples from {len(specimens)} specimens.")
    return np.array(X_list), np.array(y_list), specimen_keys


def train_and_evaluate():
    """
    Executes the main pipeline for training and evaluating the RNN model.
    This includes:
    1.  Creating necessary directories for models and reports.
    2.  Loading and preparing data using `create_sequence_samples`.
    3.  Splitting data into training and testing sets.
    4.  Scaling features and targets using MinMaxScaler.
    5.  Building the LSTM-based RNN model.
    6.  Training the model.
    7.  Saving the trained model and scalers.
    8.  Evaluating the model's performance on the test set.
    9.  Saving an evaluation report.
    """
    # Create directories if they don't exist.
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading and creating sequence samples...")
    X, y, specimen_keys = create_sequence_samples()

    if len(X) == 0:
        print("No data available after processing. Exiting.")
        return

    # Split data into training and testing sets for model evaluation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit MinMaxScaler for features.
    # Reshape for scaling, then reshape back to original 3D sequence format.
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Initialize and fit MinMaxScaler for targets.
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # --- Build the RNN Model ---
    # Define the LSTM model architecture.
    # Input shape: (sequence_length, num_features)
    # LSTM layers process sequences, Dropout layers help prevent overfitting.
    # The final Dense layer outputs the predicted target values.
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        LSTM(128, activation='tanh'),
        Dropout(0.4),
        Dense(y_train_scaled.shape[1]) # Output layer size matches flattened target
    ])

    # Compile the model with Adam optimizer and Mean Squared Error loss.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse') # Added learning rate to Adam optimizer
    model.summary() # Print a summary of the model architecture.

    print("\nTraining the RNN model...")
    # Define EarlyStopping callback to prevent overfitting.
    # It monitors 'val_loss' and stops training if it doesn't improve for 'patience' epochs.
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Train the model.
    # epochs: Number of times to iterate over the entire dataset.
    # batch_size: Number of samples per gradient update.
    # validation_split: Fraction of the training data to be used as validation data.
    # callbacks: List of callbacks to apply during training.
    # Note: With a small dataset (like 20 samples), overfitting is a significant risk.
    # Early stopping helps, but more data or advanced techniques (e.g., data augmentation)
    # would be beneficial for robust performance.
    history = model.fit(X_train_scaled, y_train_scaled, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # --- Save the Model and Scalers ---
    # Save the trained Keras model and the fitted scalers for future use.
    model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_model.keras')
    model.save(model_path)
    joblib.dump(scaler_X, os.path.join(MODELS_DIR, f'{MODEL_NAME}_scaler_X.joblib'))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, f'{MODEL_NAME}_scaler_y.joblib'))
    print(f"\nModel and scalers saved to {MODELS_DIR}")

    # --- Evaluation ---
    print("\nEvaluating the model on the test set...")
    # Make predictions on the scaled test set.
    y_pred_scaled = model.predict(X_test_scaled)
    # Inverse transform the predictions to get original scale values.
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Calculate R-squared score to evaluate model performance.
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print(f"  Test Set R² Score: {r2:.4f}")

    # --- Save Evaluation Metrics ---
    # Generate and save a detailed evaluation report.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(REPORTS_DIR, f'{MODEL_NAME}_evaluation.txt')

    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report generated on: {timestamp}\n")
        f.write(f"Model: {MODEL_NAME}\n\n")
        f.write(f"Preprocessing Parameters:\n")
        f.write(f"  FIXED_LENGTH: {FIXED_LENGTH}\n")
        f.write(f"  INPUT_SPLIT: {INPUT_SPLIT}\n\n")
        f.write(f"Final Test Set R² score: {r2:.4f}\n")

    print(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    train_and_evaluate()
    print("\nScript finished.")

