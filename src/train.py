import json
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS_FILE_PATH,
    METRICS_FILE_PATH,
    MODEL_FILE_PATH,
    RANDOM_STATE,
    SCALER_FILE_PATH,
)
from src.custom_exception import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)


def scale_data(x_train, x_test):
    try:
        logger.info("Scaling features with MinMaxScaler.")
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        logger.info("Feature scaling completed successfully.")
        return x_train_scaled, x_test_scaled, scaler
    except Exception as exc:
        logger.exception("Error occurred during feature scaling.")
        raise ProjectException("Failed to scale data.") from exc


def train_model(x_train_scaled, y_train):
    """
    Notebook-faithful model:
    MLPClassifier(
        activation='tanh',
        batch_size=50,
        hidden_layer_sizes=3,
        random_state=123
    )
    """
    try:
        logger.info("Training MLPClassifier model.")
        model = MLPClassifier(
            activation="tanh",
            batch_size=50,
            hidden_layer_sizes=3,
            max_iter=200,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train_scaled, y_train)
        logger.info("Model training completed successfully.")
        return model
    except Exception as exc:
        logger.exception("Error occurred during model training.")
        raise ProjectException("Failed to train model.") from exc


def evaluate_model(model, x_train_scaled, y_train, x_test_scaled, y_test):
    try:
        logger.info("Evaluating model.")

        y_pred_train = model.predict(x_train_scaled)
        y_pred_test = model.predict(x_test_scaled)

        metrics = {
            "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "test_confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        }

        logger.info("Model evaluation completed: %s", metrics)
        return metrics
    except Exception as exc:
        logger.exception("Error occurred during model evaluation.")
        raise ProjectException("Failed to evaluate model.") from exc


def save_artifacts(model, scaler, feature_columns, metrics):
    try:
        logger.info("Saving model artifacts.")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(MODEL_FILE_PATH, "wb") as f:
            pickle.dump(model, f)

        with open(SCALER_FILE_PATH, "wb") as f:
            pickle.dump(scaler, f)

        with open(FEATURE_COLUMNS_FILE_PATH, "wb") as f:
            pickle.dump(feature_columns, f)

        with open(METRICS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Artifacts saved successfully.")
    except Exception as exc:
        logger.exception("Error occurred while saving artifacts.")
        raise ProjectException("Failed to save artifacts.") from exc