import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

from model import detect_fraud, train_model, validate_dataset

def visualize_results_from_file(file_path):
    """
    Load results from a file and visualize them.

    Parameters:
        file_path (str): Path to the results file.
    """
    try:
        # Load results
        results = pd.read_csv(file_path)
        print("Results loaded successfully.")

        # Validate necessary columns
        if 'Class' not in results.columns or 'Fraud' not in results.columns:
            raise KeyError("Dataset must contain 'Class' and 'Fraud' columns.")

        # Count of actual vs. detected frauds
        plt.figure(figsize=(12, 6))

        # Actual fraud vs normal transactions
        plt.subplot(1, 2, 1)
        sns.countplot(x='Class', data=results, palette='viridis')
        plt.title('Actual Class Distribution')
        plt.xlabel('Class (0 = Normal, 1 = Fraud)')
        plt.ylabel('Count')

        # Predicted fraud vs normal transactions
        plt.subplot(1, 2, 2)
        sns.countplot(x='Fraud', data=results, palette='magma')
        plt.title('Predicted Fraud Distribution')
        plt.xlabel('Fraud (0 = Normal, 1 = Fraud)')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

        # Confusion matrix heatmap
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results['Class'], results['Fraud'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except KeyError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

results_path = r'C:\Users\Ibrah\OneDrive\Desktop\isolation-forest-tool\data\fraud_detection_results.csv'


# Visualize from saved results
visualize_results_from_file(results_path)



