# Fraud Detection using Isolation Forest

This project implements fraud detection using the Isolation Forest algorithm. The goal is to train a model to identify fraudulent transactions from a credit card dataset and visualize the results.

## Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv  #  dataset file
├── notebooks/
│   └── EDA.ipynb       # Jupyter notebook for Exploratory Data Analysis
├── src/
│   ├── preprocess.py   # Data preprocessing scripts
│   ├── model.py        # Training the Isolation Forest model
│   ├── visualize.py    # Visualization scripts
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation
├── LICENSE             
└── .gitignore          # To exclude unnecessary files
```

### Description

This repository contains the code and resources necessary to implement an anomaly detection model for identifying fraudulent credit card transactions. The model is trained using the Isolation Forest algorithm and the dataset is preprocessed to standardize features and remove irrelevant columns.

### Dataset

The dataset used in this project is the **Credit Card Fraud Detection dataset** available from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset includes information on credit card transactions, with a **Class** column that indicates whether the transaction is fraudulent (1) or not (0).

### How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ibrahimboutal/fraud-detection.git
   cd fraud-detection
   ```

2. **Install dependencies**:
   It's recommended to create a virtual environment and install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**:
   - Open the `notebooks/EDA.ipynb` for Exploratory Data Analysis (EDA).
   - You can train and evaluate the Isolation Forest model directly by running the scripts in the `src/` directory.

### Key Scripts

- **`src/preprocess.py`**: Contains the data preprocessing logic such as normalizing the `Amount` column and removing irrelevant features like `Time`.
  
- **`src/model.py`**: Implements the Isolation Forest model training and fraud detection.

- **`src/visualize.py`**: Includes functions to visualize the results, such as plotting the fraud detection results or feature importance.

- **`notebooks/EDA.ipynb`**: A Jupyter notebook for performing exploratory data analysis on the credit card dataset, helping to understand patterns and distribution of fraudulent transactions.

### Results

The model predicts fraud based on transaction details, and you can visualize the results with the `visualize.py` script. The output will include:
- **Fraudulent transactions** detected by the model.
- **Model evaluation** metrics such as precision, recall, and F1-score.

### Dependencies

- Python 3.12.8
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute by opening issues or submitting pull requests.
```

### Explanation:

- **Overview**: Describes the goal and purpose of the project.
- **Project Structure**: Gives an outline of the project files and folders.
- **Dataset**: Mentions the dataset used for the project and provides a link to it.
- **How to Run**: Provides steps to set up and run the project, including installing dependencies.
- **Key Scripts**: Highlights important scripts for preprocessing, modeling, and visualizing the results.
- **Results**: Describes the kind of results you can expect from the project.
- **Dependencies**: Lists the dependencies required to run the project and provides installation instructions.
- **License**: Specifies that the project is licensed under the MIT License.

This template should help users or collaborators easily understand how to use the repository and its components. You can further customize it based on the specifics of your project!