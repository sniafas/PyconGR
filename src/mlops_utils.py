import json
import boto3

import botocore.exceptions
import os
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from typing import Tuple

load_dotenv()

def put_object_to_s3(data: dict, bucket_name: str, s3_key: str):
    """
    Writes a Python dictionary to a JSON file in an S3 bucket.

    Args:
        data (dict): The dictionary to write to S3.
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The S3 key (path) for the JSON file.
            e.g., 'my-folder/data.json'
    """
    try:
        # convert the dictionary to a JSON string
        data_serialized = json.dumps(data, default=str)

        # create session with profile name from .aws/credentials
        session = boto3.session.Session(profile_name=os.getenv("S3_PROFILE_NAME"))

        # Create an S3 client
        s3_client = session.client("s3")

        # put json object to s3 bucket
        response = s3_client.put_object(
            Body=data_serialized,
            Bucket=bucket_name,
            Key=s3_key,
            ContentType="application/json",
        )

        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            logger.info(f"Successfully wrote JSON to s3://{bucket_name}/{s3_key}")
        else:
            logger.error(f"Failed to write JSON to s3://{bucket_name}/{s3_key}.  Response: {response}")

    except botocore.exceptions.ClientError as e:
        logger.error(f"Error write data to S3: {e}")
        return None


def load_object_from_s3(bucket_name: str, s3_key: str) -> dict:
    """Load object from s3

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The S3 key (path) for the JSON file.
            e.g., 'my-folder/data.json'

    Returns:
        dict: data
    """
    try:
        # create session with profile name from .aws/credentials
        session = boto3.session.Session(profile_name=os.getenv("S3_PROFILE_NAME"))

        # create an S3 client
        s3_client = session.client("s3")

        # read object
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=s3_key,
        )

        # read and process the data
        data = json.loads(response["Body"].read().decode("utf-8"))

    except botocore.exceptions.ClientError as e:
        logger.error(f"Error loading data from S3 Outposts: {e}")
        return None

    return data


def load_tracked_artifacts() -> Tuple[dict]:
    """Load tracked model and data artifacts from MLflow server. It returns two dicts of models and data with s3 URIs.

    Returns:
        Tuple[dict, dict]: model_list, artifact_list
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlf_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlf_tracking_uri)
    # create an mlflow client
    client = MlflowClient()

    # Search for all registered models
    registered_models = client.search_registered_models("name='iris_model'")

    model_list = {}
    for reg_model in registered_models:

            # get model uri
            model_uri = reg_model.latest_versions[0].source

            # get model name
            model_name = reg_model.name

            # get model version
            model_version = reg_model.latest_versions[0].version

            # create model uri
            model_relative_uri = f"models:/{reg_model.name}/{model_version}"

            # create model registered date from unix timestamp
            model_registered_date = datetime.utcfromtimestamp(
                int(str(reg_model.last_updated_timestamp)[:10])
            ).strftime("%Y%m%d")

            # append them to model list
            model_list[model_name] = {
                "path": model_uri,
                "model_relative_uri": model_relative_uri,
                "version": model_version,
                "date": model_registered_date,
                "model": "", # this serves as the binary placeholder
            }

    return model_list


