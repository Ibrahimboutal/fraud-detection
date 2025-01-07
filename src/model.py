import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import classification_report

def validate_dataset(data):
    """
    Validate the dataset to ensure it contains the required columns.

    Parameters:
        data (pd.DataFrame): Dataset to validate.

    Returns:
        bool: True if the dataset is valid, False otherwise.
    """
    required_columns = {'Class'}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    return True

def train_model(data, n_estimators=100, contamination=0.01, random_state=42):
    """
    Train an Isolation Forest model on the dataset.

    Parameters:
        data (pd.DataFrame): Preprocessed dataset.
        n_estimators (int): Number of trees in the forest.
        contamination (float): Proportion of outliers in the dataset.
        random_state (int): Random seed.

    Returns:
        IsolationForest: Trained model.
    """
    try:
        # Validate dataset
        if 'Class' not in data.columns:
            raise KeyError("Column 'Class' is missing from the dataset.")

        # Define and fit the model
        model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=random_state
        )
        X = data.drop(['Class'], axis=1)
        model.fit(X)
        print("Model training completed successfully.")
        return model
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None

def save_results(data, file_path):
    """
    Save the dataset with fraud detection results to a file.

    Parameters:
        data (pd.DataFrame): Dataset containing fraud detection results.
        file_path (str): Path to save the results.
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Results successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving results: {e}")

def detect_fraud(model, data):
    """
    Detect fraudulent transactions using the trained Isolation Forest model.

    Parameters:
        model (IsolationForest): Trained Isolation Forest model.
        data (pd.DataFrame): Preprocessed dataset.

    Returns:
        pd.DataFrame: Dataset with an additional 'Fraud' column.
    """
    try:
        if model is None:
            raise ValueError("Model is not provided.")

        if 'Class' not in data.columns:
            raise KeyError("Column 'Class' is missing from the dataset.")

        # Predict and assign fraud labels
        predictions = model.predict(data.drop(['Class'], axis=1))
        data['Fraud'] = np.where(predictions == -1, 1, 0)
        print("Fraud detection completed successfully.")
        return data
    except Exception as e:
        print(f"An error occurred during fraud detection: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # File paths
    preprocessed_data_path = r'C:\Users\Ibrah\OneDrive\Desktop\isolation-forest-tool\data\processed_creditcard.csv'
    results_path = r'C:\Users\Ibrah\OneDrive\Desktop\isolation-forest-tool\data\fraud_detection_results.csv'


    # Load the preprocessed data
    try:
        processed_data = pd.read_csv(preprocessed_data_path)
        print("Processed data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {preprocessed_data_path}")
        processed_data = None

    # Ensure the dataset is valid and proceed with model training and detection
    if processed_data is not None and validate_dataset(processed_data):
        isolation_forest_model = train_model(processed_data)
        if isolation_forest_model:
            results = detect_fraud(isolation_forest_model, processed_data)
            if results is not None:
                save_results(results, results_path)
                # Display results and evaluate model performance
                print(results.head())
                print("\nClassification Report:")
                print(classification_report(results['Class'], results['Fraud']))
