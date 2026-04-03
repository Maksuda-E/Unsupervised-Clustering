# Import pandas for dataframe operations.
import pandas as pd

# Import KMeans for clustering.
from sklearn.cluster import KMeans

# Import all required configuration classes.
from src.config import ArtifactConfig, DataConfig, ModelConfig

# Import the custom exception class.
from src.custom_exception import ProjectException

# Import the dataset loader.
from src.data_loader import DataLoader

# Import the evaluator class.
from src.evaluate import ClusterEvaluator

# Import the project logger.
from src.logger import logger

# Import the preprocessor class.
from src.preprocessing import DataPreprocessor

# Import utility functions for saving files.
from src.utils import ensure_directories, save_json, save_object


# Create a function to return simple notebook based cluster names.
def get_segment_name(cluster_id: int) -> str:
    # Map cluster id to notebook style segment name.
    mapping = {
        0: "Average customers",
        1: "High income low spending",
        2: "High income high spending",
        3: "Low income high spending",
        4: "Low income low spending",
    }

    # Return the cluster name if found.
    return mapping.get(cluster_id, f"Cluster {cluster_id}")


# Create a function to return simple notebook based meanings.
def get_segment_meaning(cluster_id: int) -> str:
    # Map cluster id to a short human readable meaning.
    mapping = {
        0: "Customers with average income and average spending.",
        1: "Customers who earn a lot but spend less.",
        2: "Customers who earn a lot and spend a lot.",
        3: "Customers who earn less but spend more.",
        4: "Customers who earn less and spend less.",
    }

    # Return the meaning if found.
    return mapping.get(cluster_id, "General customer group.")


# Create a function to build the segment profile dictionary.
def create_segment_profiles(cluster_profile_df: pd.DataFrame) -> dict:
    # Create an empty dictionary for all profiles.
    profiles = {}

    # Loop through each row in the cluster summary table.
    for _, row in cluster_profile_df.iterrows():
        # Convert the cluster value to integer.
        cluster_id = int(row["Cluster"])

        # Create the dictionary for the current cluster.
        profiles[str(cluster_id)] = {
            "segment_name": get_segment_name(cluster_id),
            "meaning": get_segment_meaning(cluster_id),
        }

    # Return the complete profiles dictionary.
    return profiles


# Create the full training pipeline function.
def run_training_pipeline() -> None:
    # Start a try block.
    try:
        # Create artifacts and log folders if needed.
        ensure_directories(ArtifactConfig.ARTIFACT_DIR, ArtifactConfig.LOG_DIR)

        # Log the start of training.
        logger.info("Training pipeline started")

        # Create a loader object.
        data_loader = DataLoader(DataConfig.DATA_PATH)

        # Load the dataset.
        df = data_loader.load_data()

        # Create the preprocessor using the final project feature columns.
        preprocessor = DataPreprocessor(ModelConfig.FEATURE_COLUMNS)

        # Scale the training features.
        X_scaled = preprocessor.fit_transform(df)

        # Create the evaluator object.
        evaluator = ClusterEvaluator(
            random_state=ModelConfig.RANDOM_STATE,
            init=ModelConfig.INIT,
            n_init=ModelConfig.N_INIT,
            max_iter=ModelConfig.MAX_ITER,
        )

        # Create the notebook style cluster search range.
        k_range = range(3, 9)

        # Calculate elbow scores.
        elbow_df = evaluator.calculate_elbow(X_scaled, k_range)

        # Calculate silhouette scores.
        silhouette_df = evaluator.calculate_silhouette(X_scaled, k_range)

        # Save elbow scores to CSV.
        elbow_df.to_csv(ArtifactConfig.ELBOW_PATH, index=False)

        # Save silhouette scores to CSV.
        silhouette_df.to_csv(ArtifactConfig.SILHOUETTE_PATH, index=False)

        # Create the final KMeans model with 5 clusters based on notebook segment type.
        model = KMeans(
            n_clusters=ModelConfig.N_CLUSTERS,
            random_state=ModelConfig.RANDOM_STATE,
            init=ModelConfig.INIT,
            n_init=ModelConfig.N_INIT,
            max_iter=ModelConfig.MAX_ITER,
        )

        # Fit the model and get cluster labels.
        cluster_labels = model.fit_predict(X_scaled)

        # Store labels back into the dataframe.
        df["Cluster"] = cluster_labels

        # Build cluster wise average profile table.
        cluster_profile_df = (
            df.groupby("Cluster")[ModelConfig.FEATURE_COLUMNS]
            .mean()
            .round(2)
            .reset_index()
        )

        # Add customer count per cluster.
        cluster_profile_df["Customer_Count"] = df["Cluster"].value_counts().sort_index().values

        # Build the segment meaning dictionary.
        segment_profiles = create_segment_profiles(cluster_profile_df)

        # Save clustered training data.
        df.to_csv(ArtifactConfig.CLUSTERED_DATA_PATH, index=False)

        # Save cluster average profile table.
        cluster_profile_df.to_csv(ArtifactConfig.CLUSTER_PROFILE_PATH, index=False)

        # Save the segment JSON file.
        save_json(ArtifactConfig.SEGMENT_PROFILE_PATH, segment_profiles)

        # Save the trained model file.
        save_object(ArtifactConfig.MODEL_PATH, model)

        # Save the fitted scaler file.
        save_object(ArtifactConfig.SCALER_PATH, preprocessor.scaler)

        # Log successful completion.
        logger.info("Training pipeline completed successfully")

        # Print success messages for the terminal.
        print("Training complete")
        print(f"Model saved to: {ArtifactConfig.MODEL_PATH}")
        print(f"Scaler saved to: {ArtifactConfig.SCALER_PATH}")
        print(f"Cluster profiles saved to: {ArtifactConfig.CLUSTER_PROFILE_PATH}")
        print(f"Segment profiles saved to: {ArtifactConfig.SEGMENT_PROFILE_PATH}")

    # Catch any pipeline error.
    except Exception as exc:
        # Log the full exception.
        logger.exception("Training pipeline failed")

        # Raise a project specific exception.
        raise ProjectException(f"Training pipeline failed: {exc}") from exc