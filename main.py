# This line imports the dataset file path
from src.config import DATA_FILE_PATH

# This line imports the data loading function
from src.data_loader import load_data

# This line imports preprocessing functions
from src.preprocess import clean_data, select_features

# This line imports training functions
from src.train import train_model, save_artifacts

# This line imports the logger
from src.logger import get_logger

# This line creates a logger for this file
logger = get_logger(__name__)

# This function controls the full training pipeline
def main():
    # This line starts the try block
    try:
        # This line logs pipeline start
        logger.info("Main pipeline started")

        # This line loads the dataset
        df = load_data(DATA_FILE_PATH)

        # This line cleans the dataset
        df_clean = clean_data(df)

        # This line selects the clustering features
        x = select_features(df_clean)

        # This line trains the clustering model
        model, scaler, metrics, cluster_mapping = train_model(x)

        # This line saves the artifacts
        save_artifacts(model, scaler, metrics, cluster_mapping)

        # This line prints success message
        print("Training completed successfully")

        # This line prints the metrics heading
        print("Model evaluation results")

        # This line loops through metrics
        for key, value in metrics.items():
            # This line prints each metric
            print(f"{key}: {value}")

        # This line logs pipeline completion
        logger.info("Main pipeline completed successfully")

    # This block handles pipeline errors
    except Exception as exc:
        # This line logs pipeline failure
        logger.error(f"Main pipeline failed: {exc}")

        # This line raises the error again
        raise

# This line checks if the file is run directly
if __name__ == "__main__":
    # This line calls the main function
    main()
