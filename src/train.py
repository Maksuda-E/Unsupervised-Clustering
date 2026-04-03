# Import pandas for dataframe operations.
import pandas as pd

# Import KMeans for clustering.
from sklearn.cluster import KMeans

# Import configuration classes.
from src.config import ArtifactConfig, DataConfig, ModelConfig

# Import custom exception class.
from src.custom_exception import ProjectException

# Import data loader.
from src.data_loader import DataLoader

# Import evaluator.
from src.evaluate import ClusterEvaluator

# Import logger.
from src.logger import logger

# Import preprocessor.
from src.preprocessing import DataPreprocessor

# Import utility functions.
from src.utils import ensure_directories, save_json, save_object


# Return segment name based on cluster profile values.
def get_segment_name(age: float, income: float, spending: float) -> str:
    # High income and high spending customers.
    if income >= 70 and spending >= 70:
        return "High income high spending"

    # High income and low spending customers.
    if income >= 70 and spending < 40:
        return "High income low spending"

    # Low income and high spending customers.
    if income < 40 and spending >= 60:
        return "Low income high spending"

    # Low income and low spending customers.
    if income < 40 and spending < 40:
        return "Low income low spending"

    # Younger balanced customers.
    if age < 35:
        return "Young average customers"

    # Older balanced customers.
    return "Older average customers"


# Return simple meaning based on cluster profile values.
def get_segment_meaning(age: float, income: float, spending: float) -> str:
    # High income and high spending customers.
    if income >= 70 and spending >= 70:
        return "Customers who earn a lot and spend a lot."

    # High income and low spending customers.
    if income >= 70 and spending < 40:
        return "Customers who earn a lot but spend less."

    # Low income and high spending customers.
    if income < 40 and spending >= 60:
        return "Customers who earn less but still spend more."

    # Low income and low spending customers.
    if income < 40 and spending < 40:
        return "Customers who earn less and spend less."

    # Younger balanced customers.
    if age < 35:
        return "Younger customers with medium income and medium spending."

    # Older balanced customers.
    return "Older customers with medium income and medium spending."


# Build the segment profile dictionary.
def create_segment_profiles(cluster_profile_df: pd.DataFrame) -> dict:
    # Create an empty dictionary.
    profiles = {}

    # Loop through each cluster row.
    for _, row in cluster_profile_df.iterrows():
        # Extract the cluster id.
        cluster_id = int(row["Cluster"])

        # Extract cluster averages.
        age = float(row["Age"])
        income = float(row["Annual_Income"])
        spending = float(row["Spending_Score"])

        # Save segment information for this cluster.
        profiles[str(cluster_id)] = {
            "segment_name": get_segment_name(age, income, spending),
            "meaning": get_segment_meaning(age, income, spending),
        }

    # Return the full dictionary.
    return profiles


# Run the complete training pipeline.
def run_training_pipeline() -> None:
    # Start a safe try block.
    try:
        # Create required folders.
        ensure_directories(ArtifactConfig.ARTIFACT_DIR, ArtifactConfig.LOG_DIR)

        # Log pipeline start.
        logger.info("Training pipeline started")

        # Load the dataset.
        data_loader = DataLoader(DataConfig.DATA_PATH)
        df = data_loader.load_data()

        # Preprocess the selected features.
        preprocessor = DataPreprocessor(ModelConfig.FEATURE_COLUMNS)
        X_scaled = preprocessor.fit_transform(df)

        # Create the evaluator object.
        evaluator = ClusterEvaluator(
            random_state=ModelConfig.RANDOM_STATE,
            init=ModelConfig.INIT,
            n_init=ModelConfig.N_INIT,
            max_iter=ModelConfig.MAX_ITER,
        )

        # Define the k range used in the notebook.
        k_range = range(3, 9)

        # Calculate elbow scores.
        elbow_df = evaluator.calculate_elbow(X_scaled, k_range)

        # Calculate silhouette scores.
        silhouette_df = evaluator.calculate_silhouette(X_scaled, k_range)

        # Save elbow scores.
        elbow_df.to_csv(ArtifactConfig.ELBOW_PATH, index=False)

        # Save silhouette scores.
        silhouette_df.to_csv(ArtifactConfig.SILHOUETTE_PATH, index=False)

        # Create the final KMeans model with 6 clusters.
        model = KMeans(
            n_clusters=ModelConfig.N_CLUSTERS,
            random_state=ModelConfig.RANDOM_STATE,
            init=ModelConfig.INIT,
            n_init=ModelConfig.N_INIT,
            max_iter=ModelConfig.MAX_ITER,
        )

        # Fit the model and create cluster labels.
        cluster_labels = model.fit_predict(X_scaled)

        # Store cluster labels in the dataframe.
        df["Cluster"] = cluster_labels

        # Build the cluster summary table.
        cluster_profile_df = (
            df.groupby("Cluster")[ModelConfig.FEATURE_COLUMNS]
            .mean()
            .round(2)
            .reset_index()
        )

        # Add customer count per cluster.
        cluster_profile_df["Customer_Count"] = df["Cluster"].value_counts().sort_index().values

        # Build simple segment profiles.
        segment_profiles = create_segment_profiles(cluster_profile_df)

        # Save clustered training data.
        df.to_csv(ArtifactConfig.CLUSTERED_DATA_PATH, index=False)

        # Save cluster profile table.
        cluster_profile_df.to_csv(ArtifactConfig.CLUSTER_PROFILE_PATH, index=False)

        # Save segment profile json.
        save_json(ArtifactConfig.SEGMENT_PROFILE_PATH, segment_profiles)

        # Save model object.
        save_object(ArtifactConfig.MODEL_PATH, model)

        # Save scaler object.
        save_object(ArtifactConfig.SCALER_PATH, preprocessor.scaler)

        # Log success.
        logger.info("Training pipeline completed successfully")

        # Print success messages.
        print("Training complete")
        print(f"Model saved to: {ArtifactConfig.MODEL_PATH}")
        print(f"Scaler saved to: {ArtifactConfig.SCALER_PATH}")

    # Catch any error.
    except Exception as exc:
        # Log the full exception.
        logger.exception("Training pipeline failed")

        # Raise a project specific exception.
        raise ProjectException(f"Training pipeline failed: {exc}") from exc