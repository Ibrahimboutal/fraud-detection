import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the dataset by normalizing features and removing unnecessary columns.

    Parameters:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    if data is None:
        print("Error: No data provided for preprocessing.")
        return None

    try:
        # Normalize the 'Amount' column
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

        # Drop the 'Time' column if it exists
        if 'Time' in data.columns:
            data = data.drop(['Time'], axis=1)

        print("Data preprocessing completed successfully.")
        return data
    except KeyError as e:
        print(f"Error: Missing expected column: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None

# File path
file_path = r'C:\Users\Ibrah\OneDrive\Desktop\isolation-forest-tool\data\creditcard.csv'

# Main execution
if __name__ == "__main__":
    # Load data
    raw_data = load_data(file_path)
    if raw_data is not None:
        # Preprocess data
        processed_data = preprocess_data(raw_data)

        # Display processed data
        if processed_data is not None:
            print(processed_data.head())
            processed_data.to_csv(r'C:\Users\Ibrah\OneDrive\Desktop\isolation-forest-tool\data\processed_creditcard.csv', index=False)
print("Processed data saved successfully.")

