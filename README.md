# Iris Linear Regression with MLflow

This project demonstrates training a Linear Regression model on the Iris dataset, logging the experiment and its artifacts using MLflow.

## Prerequisites

1. **Python 3.x**  
2. **MLflow**  
3. **Pandas**  
4. **Scikit-learn**  
5. **Matplotlib**  
6. **Kagglehub** (for dataset download)

## Installation
```bash
pip install mlflow scikit-learn pandas matplotlib kagglehub
```

#### Download the Iris dataset from Kaggle
```bash
# Download the dataset
path = kagglehub.dataset_download("uciml/iris")
print("Path to dataset files:", path)

# Read the actual CSV file
csv_path = os.path.join(path, "Iris.csv")
print("Reading CSV file from:", csv_path)
df = pd.read_csv(csv_path)
```

## Run Model 
```bash
python train_model.py    # can run in VS Code terminal
```

## Run MLFlow UI
```bash
mlflow ui                # run on command shell
```
-- Click on the url printed in shell to open dashboard. 

## Code Explanation
Data Preprocessing: Reads the Iris dataset, drops missing values, and selects features for the model.
Model Training: Fits a Linear Regression model on the training data.
MLflow Logging: Logs the experiment, including metrics, parameters, and a plot comparing actual vs predicted values.
MLflow UI: Provides a dashboard to monitor and compare multiple experiments.
