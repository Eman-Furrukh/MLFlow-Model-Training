import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import kagglehub

# Download the dataset
path = kagglehub.dataset_download("uciml/iris")
print("Path to dataset files:", path)

# Read the actual CSV file
csv_path = os.path.join(path, "Iris.csv")
print("Reading CSV file from:", csv_path)
df = pd.read_csv(csv_path)

# Drop any missing values (safety)
df.dropna(inplace=True)

# Drop 'Id' and 'Species' columns since we're doing regression
X = df[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['SepalLengthCm']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Iris_LinearRegression"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    print(f"MSE: {mse}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, preds)
    plt.xlabel("Actual Sepal Length")
    plt.ylabel("Predicted Sepal Length")
    plt.title("Actual vs Predicted Sepal Length")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    mlflow.log_artifact("actual_vs_predicted.png")
