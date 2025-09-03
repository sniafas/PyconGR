# Hot-Swapping MLflow Models on AWS Lambda

This repository contains the demo code for a PyCon talk on hot-swapping MLflow machine learning models on AWS Lambda with zero downtime. The solution leverages Python's global variables and a simple time-scheduled check to dynamically update models in a serverless environment without requiring a full Lambda redeployment.
Project Architecture

The core of this project consists of the following components:
 - MLflow Model Training (train_model.py): A script that trains a simple regression model on the Iris dataset, and then logs and registers it with MLflow. This model is then packaged for deployment.
 - Model serialization (serialize_models.py): Use MLflow Client to parse register models and boto3 to upload a serialized list to s3
 - AWS Lambda Function (lambda_function.py): The serverless component that performs the following actions on every invocation and implements a time-based check to periodically check for new model versions in S3.
 - HTTP Client (client_script.py): A client script that continuously calls the Lambda function every second to simulate real-world traffic and demonstrate the zero-downtime model update.

# Demo Walkthrough
1. Train and Log the Initial Model

```
python train_model.py
python serialize_models.py
python client.py
```
This will create an mlruns directory. Inside, you'll find your packaged MLflow model.

2. Deploy to AWS S3 and Lambda

        Enable a Lambda Function URL for the function to get an HTTP endpoint.

3. Run the Zero-Downtime Demo
With your Lambda deployed and running, you can now execute the client script.
