import os
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

from mlops_utils import load_tracked_artifacts, put_object_to_s3

# load environment variables from .env file
load_dotenv()

model_list = load_tracked_artifacts()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_KEY_PATH_BASE = Path("pycon/")

for model_name in model_list.keys():
    if model_list[model_name].get("path").startswith("models:/"):
        src_model_name = "_".join(model_list[model_name].get("path").split("/")[1]).upper()
        src_model_path = model_list[src_model_name].get("path")
        
        # {model_name: model_path}
        model_list[model_name.upper()].update({"path": src_model_path})

MODEL_LIST_PATH = str(S3_KEY_PATH_BASE / "model_list.json")

logger.info(model_list)
put_object_to_s3(model_list, S3_BUCKET_NAME, MODEL_LIST_PATH)
