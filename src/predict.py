# This line imports pickle for loading saved objects
import pickle

# This line imports pandas for creating input DataFrame
import pandas as pd

# This line imports model and scaler paths
from src.config import MODEL_FILE_PATH, SCALER_FILE_PATH

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function loads the saved model and scaler
def load_model_and_scaler():
    # This line starts the try block
    try:
        # This line opens the model file
        with open(MODEL_FILE_PATH, "rb") as model_file:
            model = pickle.load(model_file)

        # This line opens the scaler file
        with open(SCALER_FILE_PATH, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # This line logs successful loading
        logger.info("Model and scaler loaded successfully")

        # This line returns the loaded objects
        return model, scaler

    # This block handles loading errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while loading model or scaler")

        # This line raises a custom exception
        raise ProjectException(f"Failed to load model artifacts: {exc}")

# This function prepares user input for prediction
def prepare_input_data(user_input: dict):
    # This line starts the try block
    try:
        # This line creates a DataFrame from user input
        input_df = pd.DataFrame([user_input])

        # This line returns the prepared input
        return input_df

    # This block handles preparation errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while preparing input data")

        # This line raises a custom exception
        raise ProjectException(f"Failed to prepare input data: {exc}")

# This function predicts the cluster
def predict_cluster(user_input: dict):
    # This line starts the try block
    try:
        # This line loads the model and scaler
        model, scaler = load_model_and_scaler()

        # This line prepares input data
        input_df = prepare_input_data(user_input)

        # This line scales the input data
        input_scaled = scaler.transform(input_df)

        # This line predicts the cluster
        cluster = model.predict(input_scaled)[0]

        # This line logs prediction success
        logger.info("Cluster prediction completed successfully")

        # This line returns the cluster as integer
        return int(cluster)

    # This block handles prediction errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during cluster prediction")

        # This line raises a custom exception
        raise ProjectException(f"Failed to predict cluster: {exc}")