# This line imports os for folder creation
import os

# This line imports json for saving metrics
import json

# This line imports pickle for saving model objects
import pickle

# This line imports KMeans for clustering
from sklearn.cluster import KMeans

# This line imports StandardScaler for scaling features
from sklearn.preprocessing import StandardScaler

# This line imports silhouette_score for evaluation
from sklearn.metrics import silhouette_score

# This line imports configuration values
from src.config import ARTIFACTS_DIR, MODEL_FILE_PATH, SCALER_FILE_PATH, METRICS_FILE_PATH, RANDOM_STATE, N_CLUSTERS

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function scales the input features
def scale_features(x):
    # This line starts the try block
    try:
        # This line logs scaling start
        logger.info("Feature scaling started")

        # This line creates the scaler
        scaler = StandardScaler()

        # This line fits and transforms the feature data
        x_scaled = scaler.fit_transform(x)

        # This line logs scaling completion
        logger.info("Feature scaling completed successfully")

        # This line returns the scaled data and scaler
        return x_scaled, scaler

    # This block handles scaling errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during feature scaling")

        # This line raises a custom exception
        raise ProjectException(f"Failed to scale features: {exc}")

# This function trains the clustering model
def train_model(x):
    # This line starts the try block
    try:
        # This line scales the data
        x_scaled, scaler = scale_features(x)

        # This line logs training start
        logger.info("KMeans model training started")

        # This line creates the KMeans model
        model = KMeans(n_clusters=N_CLUSTERS, init="k-means++", random_state=RANDOM_STATE, n_init=10)

        # This line fits the model on scaled data
        model.fit(x_scaled)

        # This line gets the cluster labels
        labels = model.labels_

        # This line calculates the silhouette score
        silhouette = silhouette_score(x_scaled, labels)

        # This line creates metrics dictionary
        metrics = {
            "n_clusters": N_CLUSTERS,
            "silhouette_score": float(silhouette),
            "inertia": float(model.inertia_)
        }

        # This line logs training completion
        logger.info("KMeans model training completed successfully")

        # This line returns the trained model, scaler, and metrics
        return model, scaler, metrics

    # This block handles training errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during model training")

        # This line raises a custom exception
        raise ProjectException(f"Failed to train clustering model: {exc}")

# This function saves the model artifacts
def save_artifacts(model, scaler, metrics):
    # This line starts the try block
    try:
        # This line creates the artifacts folder if it does not exist
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # This line opens the model file in write binary mode
        with open(MODEL_FILE_PATH, "wb") as model_file:
            # This line saves the model
            pickle.dump(model, model_file)

        # This line opens the scaler file in write binary mode
        with open(SCALER_FILE_PATH, "wb") as scaler_file:
            # This line saves the scaler
            pickle.dump(scaler, scaler_file)

        # This line opens the metrics file in write mode
        with open(METRICS_FILE_PATH, "w", encoding="utf-8") as metrics_file:
            # This line saves the metrics
            json.dump(metrics, metrics_file, indent=4)

        # This line logs successful artifact saving
        logger.info("Artifacts saved successfully")

    # This block handles saving errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while saving artifacts")

        # This line raises a custom exception
        raise ProjectException(f"Failed to save artifacts: {exc}")