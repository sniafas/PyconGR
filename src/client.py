
import time
import json
import boto3
import os
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
# Replace with your Lambda Function URL
LAMBDA_NAME = os.getenv("LAMBDA_NAME")

# The data format for our Iris regression model
# The features are sepal length, sepal width, and petal width
demo_data = {
    "data": [
        [5.1, 3.5, 0.2], 
        [6.0, 2.7, 1.6],
        [7.7, 3.0, 2.2]
    ]
}

# if you are using a custom profile name within your .aws/credentials file, change accordingly
session = boto3.Session(profile_name="ecr-profile", region_name='eu-west-1')

# create a Lambda client from this session
lambda_client = session.client('lambda')   

def invoke_lambda_post(function_name, payload_data):
    """
    Invokes an AWS Lambda function with a POST-like request payload.

    Args:
        function_name (str): The name of your Lambda function.
        payload_data (dict): A dictionary representing the data you want to send
                             in the POST request.
        region_name (str): The AWS region where your Lambda function is deployed.

    Returns:
        dict: The response from the Lambda function.
    """
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload_data)
        )

        # Read the payload from the response
        response_payload = json.loads(response['Payload'].read().decode('utf-8'))
        return response_payload

    except Exception as e:
        logger.info(f"Error invoking Lambda function: {e}")
        return None




if __name__ == "__main__":
    logger.info("Starting client. Press Ctrl+C to stop.\n")
    logger.info("To demonstrate zero-downtime, upload a new model to S3 while this script is running.")
    # import mlflow
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # model = mlflow.sklearn.load_model("mlflow-artifacts:/1/7e224b4b743f4f848ec34ae09cc66ac2/artifacts/iris_regression_model")
    # prediction = model.predict(demo_data["data"])
    # logger.info(prediction)

    while True:
        lambda_response = invoke_lambda_post(LAMBDA_NAME, demo_data)
        prediction = lambda_response.get('body').get('prediction')
        version = lambda_response.get('body').get('model_version')
        
        logger.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"  Prediction: {prediction}")
        logger.info(f"  Model Version: {version}\n")        
        time.sleep(1) # Call every 1 second
