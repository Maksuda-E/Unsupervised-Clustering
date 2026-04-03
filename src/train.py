#  imports os for folder creation
import os

#  imports json for saving metrics and cluster mapping
import json

#  imports pickle for saving model objects
import pickle

#  imports KMeans for clustering
from sklearn.cluster import KMeans

#  imports StandardScaler for scaling features
from sklearn.preprocessing import StandardScaler

#  imports silhouette_score for evaluation
from sklearn.metrics import silhouette_score

#  imports configuration values
from src.config import (
    ARTIFACTS_DIR,
    MODEL_FILE_PATH,
    SCALER_FILE_PATH,
    METRICS_FILE_PATH,
    RANDOM_STATE,
    N_CLUSTERS
)

#  imports the logger
from src.logger import get_logger

#  imports the custom exception
from src.custom_exception import ProjectException

#  creates a logger for this file
logger = get_logger(__name__)

#  defines the cluster mapping file path
CLUSTER_MAPPING_FILE_PATH = os.path.join(ARTIFACTS_DIR, "cluster_mapping.json")

# This function scales the input features
def scale_features(x):
    try:
        logger.info("Feature scaling started")

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        logger.info("Feature scaling completed successfully")
        return x_scaled, scaler

    except Exception as exc:
        logger.error("Error occurred during feature scaling")
        raise ProjectException(f"Failed to scale features: {exc}")

# This function creates readable names for each cluster from cluster centers
def create_cluster_mapping(cluster_centers):
    try:
        mapping = {}

        for idx, center in enumerate(cluster_centers):
            age, income, spending = center

            if income >= 80 and spending >= 70:
                label = "High Income High Spending Customers"
            elif income >= 80 and spending < 40:
                label = "High Income Low Spending Customers"
            elif income < 40 and spending >= 60:
                label = "Low Income High Spending Customers"
            elif income < 40 and spending < 40:
                label = "Low Income Low Spending Customers"
            elif age >= 50:
                label = "Older Moderate Customers"
            else:
                label = "Balanced Customers"

            mapping[idx] = label

        return mapping

    except Exception as exc:
        logger.error("Error occurred while creating cluster mapping")
        raise ProjectException(f"Failed to create cluster mapping: {exc}")

# This function trains the clustering model
def train_model(x):
    try:
        x_scaled, scaler = scale_features(x)

        logger.info("KMeans model training started")

        model = KMeans(
            n_clusters=N_CLUSTERS,
            init="k-means++",
            random_state=RANDOM_STATE,
            n_init=10
        )

        model.fit(x_scaled)

        labels = model.labels_
        silhouette = silhouette_score(x_scaled, labels)

        cluster_centers_original = scaler.inverse_transform(model.cluster_centers_)
        cluster_mapping = create_cluster_mapping(cluster_centers_original)

        metrics = {
            "n_clusters": N_CLUSTERS,
            "silhouette_score": float(silhouette),
            "inertia": float(model.inertia_)
        }

        logger.info("KMeans model training completed successfully")

        return model, scaler, metrics, cluster_mapping

    except Exception as exc:
        logger.error("Error occurred during model training")
        raise ProjectException(f"Failed to train clustering model: {exc}")

# This function saves the model artifacts
def save_artifacts(model, scaler, metrics, cluster_mapping):
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        with open(MODEL_FILE_PATH, "wb") as model_file:
            pickle.dump(model, model_file)

        with open(SCALER_FILE_PATH, "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)

        with open(METRICS_FILE_PATH, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)

        with open(CLUSTER_MAPPING_FILE_PATH, "w", encoding="utf-8") as mapping_file:
            json.dump(cluster_mapping, mapping_file, indent=4)

        logger.info("Artifacts saved successfully")

    except Exception as exc:
        logger.error("Error occurred while saving artifacts")
        raise ProjectException(f"Failed to save artifacts: {exc}")
