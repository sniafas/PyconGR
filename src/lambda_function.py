import json
import mlflow
import os
from loguru import logger
from datetime import datetime
from pathlib import Path

from src.mlops_utils import load_object_from_s3


S3_KEY_PATH_BASE = Path("pycon/")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MODEL_LIST_PATH = str(S3_KEY_PATH_BASE / "model_list.json")

model_list =  None
local_timestamp = -1


def load_model_artifacts():
    global model_list

    if model_list is None:
        try:
            #'iris_model': {
            #   'path': 'mlflow-artifacts:/1/7e224b4b743f4f848ec34ae09cc66ac2/artifacts/iris_regression_model',
            #   'model_relative_uri': 'models:/iris_model/2',
            #   'version': '2',
            #  'date': '20250830', 'model': ''}
            #
            model_list = load_object_from_s3(S3_BUCKET_NAME, MODEL_LIST_PATH)
            logger.info(f"Model list loaded from {MODEL_LIST_PATH}...")
        except Exception as e:
            logger.error(f"Error loading model list from S3: {e}")
            raise

        logger.info(f"Load models from s3: {model_list}")

        return model_list


def refresh_model():
    global model_list, local_timestamp

    # get the current time
    now = datetime.now()

    # round the time to the minute
    current_time = now.minute

    # check for updates every 1 minutes
    if current_time - local_timestamp > 1:
        new_model_list = load_object_from_s3(S3_BUCKET_NAME, MODEL_LIST_PATH)
        local_timestamp = current_time
        # model owners are same
        if list(model_list.keys()) == list(new_model_list.keys()):
            # iterate on model owner keys
            for model_name, model_details in model_list.items():
                # old model owner version same as new
                if model_list[model_name]["version"] != new_model_list[model_name]["version"]:
                    logger.info("Refreshing models..")
                    load_model_artifacts()
                    model_list = new_model_list


def get_model(model_name):

    # check if model is loaded
    if isinstance(model_list[model_name]["model"], str):
        model_path = model_list.get(model_name).get("path")
        logger.info(f"Model for {model_name} loads for first time from {model_path}")

        model_list[model_name]["model"] = mlflow.sklearn.load_model(model_path)

    # return model for owner
    return model_list[model_name]["model"]


def handler(request, context):
    # log event
    logger.info(request)

    load_model_artifacts()
    refresh_model()

    model = get_model("iris_model")
    
    prediction = model.predict(request["data"])
    

    return {
        "statusCode": 200,
        "body": {
            "prediction": prediction.tolist(),
            "model_version": model_list["iris_model"]["version"]
        }
    }