# This line imports json for saving metrics in JSON format
import json

# This line imports evaluation metrics from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# This line imports the metrics file path
from src.config import METRICS_FILE_PATH

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function evaluates the model
def evaluate_model(model, x_test, y_test):
    # This line starts the try block
    try:
        # This line logs that evaluation has started
        logger.info("Model evaluation started")

        # This line predicts values for the test data
        predictions = model.predict(x_test)

        # This line calculates all performance metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "f1_score": float(f1_score(y_test, predictions)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
        }

        # This line opens the metrics file in write mode
        with open(METRICS_FILE_PATH, "w", encoding="utf-8") as file:
            # This line saves the metrics dictionary as JSON
            json.dump(metrics, file, indent=4)

        # This line logs that evaluation is complete
        logger.info("Model evaluation completed successfully")

        # This line returns the metrics
        return metrics

    # This block handles evaluation errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during model evaluation")

        # This line raises a custom exception
        raise ProjectException(f"Failed to evaluate model: {exc}")