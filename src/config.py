# Import os for path handling.
import os


# Create a class to store dataset settings.
class DataConfig:
    # Store the correct dataset path.
    DATA_PATH = os.path.join("data", "mall_customers.csv")


# Create a class to store model settings.
class ModelConfig:
    # Set a fixed random seed.
    RANDOM_STATE = 42

    # Use 6 clusters to match the notebook final 3 feature conclusion.
    N_CLUSTERS = 6

    # Set centroid initialization method.
    INIT = "k-means++"

    # Set the number of initializations.
    N_INIT = 20

    # Set the maximum iterations.
    MAX_ITER = 300

    # Store the final model features.
    FEATURE_COLUMNS = ["Age", "Annual_Income", "Spending_Score"]

    # Store the 2 feature notebook exploration columns.
    NOTEBOOK_SEGMENT_COLUMNS = ["Annual_Income", "Spending_Score"]


# Create a class to store artifact paths.
class ArtifactConfig:
    # Define the artifacts folder.
    ARTIFACT_DIR = "artifacts"

    # Define the logs folder.
    LOG_DIR = "logs"

    # Store the trained model path.
    MODEL_PATH = os.path.join(ARTIFACT_DIR, "kmeans_model.pkl")

    # Store the scaler path.
    SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")

    # Store the clustered training data path.
    CLUSTERED_DATA_PATH = os.path.join(ARTIFACT_DIR, "train_clustered_data.csv")

    # Store the cluster profile csv path.
    CLUSTER_PROFILE_PATH = os.path.join(ARTIFACT_DIR, "cluster_profiles.csv")

    # Store the segment profile json path.
    SEGMENT_PROFILE_PATH = os.path.join(ARTIFACT_DIR, "segment_profiles.json")

    # Store elbow scores path.
    ELBOW_PATH = os.path.join(ARTIFACT_DIR, "elbow_curve.csv")

    # Store silhouette scores path.
    SILHOUETTE_PATH = os.path.join(ARTIFACT_DIR, "silhouette_scores.csv")


# Create a class for app settings.
class AppConfig:
    # Reuse saved segment profile path.
    SEGMENT_PROFILE_PATH = ArtifactConfig.SEGMENT_PROFILE_PATH

    # Reuse clustered data path.
    CLUSTERED_DATA_PATH = ArtifactConfig.CLUSTERED_DATA_PATH