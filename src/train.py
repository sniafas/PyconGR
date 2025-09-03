from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn
import os
def train_and_log_model():
    """
    Trains a simple regression model on the Iris dataset and logs it with MLflow.
    """
    # 1. Load the Iris dataset and prepare for regression
    iris = load_iris(as_frame=True)
    X = iris.data.drop(columns=['petal length (cm)'])
    y = iris.data['petal length (cm)']
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("pycongr")

    # 2. Start an MLflow experiment run
    with mlflow.start_run(run_name="pycon-exp"):
        # Define model parameters
        params = {"fit_intercept": True}
        mlflow.log_params(params)
        
        # 3. Train the model
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        
        # 4. Evaluate the model
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        mlflow.log_metric("mse", mse)
        
        print(f"Mean Squared Error: {mse}")
        
        # 5. Log the model
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="iris_regression_model",
            registered_model_name="iris_model"
        )
        print("Model logged to MLflow.")

if __name__ == "__main__":

    # for a local file-based server, just run `mlflow server` in a terminal
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    load_dotenv()
    mlf_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlf_tracking_uri)
    train_and_log_model()
