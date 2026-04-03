#  imports pickle for loading saved objects
import pickle

#  imports json for loading cluster mapping
import json

#  imports os for file path handling
import os

#  imports pandas for creating input DataFrame
import pandas as pd

#  imports model and scaler paths
from src.config import MODEL_FILE_PATH, SCALER_FILE_PATH, ARTIFACTS_DIR

#  imports the logger
from src.logger import get_logger

#  imports the custom exception
from src.custom_exception import ProjectException

#  creates a logger for this file
logger = get_logger(__name__)

#  defines the cluster mapping file path
CLUSTER_MAPPING_FILE_PATH = os.path.join(ARTIFACTS_DIR, "cluster_mapping.json")

# This function loads the saved model and scaler
def load_model_and_scaler():
    try:
        with open(MODEL_FILE_PATH, "rb") as model_file:
            model = pickle.load(model_file)

        with open(SCALER_FILE_PATH, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        logger.info("Model and scaler loaded successfully")
        return model, scaler

    except Exception as exc:
        logger.error("Error occurred while loading model or scaler")
        raise ProjectException(f"Failed to load model artifacts: {exc}")

# This function loads the cluster mapping
def load_cluster_mapping():
    try:
        if os.path.exists(CLUSTER_MAPPING_FILE_PATH):
            with open(CLUSTER_MAPPING_FILE_PATH, "r", encoding="utf-8") as mapping_file:
                mapping = json.load(mapping_file)
            return {int(k): v for k, v in mapping.items()}
        return {}

    except Exception as exc:
        logger.error("Error occurred while loading cluster mapping")
        raise ProjectException(f"Failed to load cluster mapping: {exc}")

# This function prepares user input for prediction
def prepare_input_data(user_input: dict):
    try:
        input_df = pd.DataFrame([user_input])
        return input_df

    except Exception as exc:
        logger.error("Error occurred while preparing input data")
        raise ProjectException(f"Failed to prepare input data: {exc}")

# This function predicts the cluster
def predict_cluster(user_input: dict):
    try:
        model, scaler = load_model_and_scaler()
        input_df = prepare_input_data(user_input)
        input_scaled = scaler.transform(input_df)
        cluster = model.predict(input_scaled)[0]

        logger.info("Cluster prediction completed successfully")
        return int(cluster)

    except Exception as exc:
        logger.error("Error occurred during cluster prediction")
        raise ProjectException(f"Failed to predict cluster: {exc}")
