import pickle
import pandas as pd

from src.config import FEATURE_COLUMNS_FILE_PATH, MODEL_FILE_PATH, SCALER_FILE_PATH
from src.custom_exception import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)


def _load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts():
    try:
        logger.info("Loading model artifacts for prediction.")
        model = _load_pickle_file(MODEL_FILE_PATH)
        scaler = _load_pickle_file(SCALER_FILE_PATH)
        feature_columns = _load_pickle_file(FEATURE_COLUMNS_FILE_PATH)
        logger.info("Artifacts loaded successfully.")
        return model, scaler, feature_columns
    except FileNotFoundError as exc:
        logger.exception("Artifact file missing.")
        raise ProjectException(
            "Artifacts not found. Run training first to generate them."
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading artifacts.")
        raise ProjectException("Failed to load model artifacts.") from exc


def prepare_input_data(user_input: dict, feature_columns: list[str]) -> pd.DataFrame:
    try:
        logger.info("Preparing user input for prediction.")

        required_fields = {
            "GRE_Score",
            "TOEFL_Score",
            "University_Rating",
            "SOP",
            "LOR",
            "CGPA",
            "Research",
        }

        missing = required_fields - set(user_input.keys())
        if missing:
            raise ProjectException(f"Missing user input fields: {sorted(missing)}")

        input_df = pd.DataFrame([user_input])

        input_df["University_Rating"] = input_df["University_Rating"].astype("object")
        input_df["Research"] = input_df["Research"].astype("object")

        input_df = pd.get_dummies(
            input_df,
            columns=["University_Rating", "Research"],
            dtype=int,
        )

        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        logger.info("Input prepared successfully.")
        return input_df

    except ProjectException:
        logger.exception("ProjectException while preparing input.")
        raise
    except Exception as exc:
        logger.exception("Unexpected error while preparing input.")
        raise ProjectException("Failed to prepare input data.") from exc


def predict_admission(user_input: dict) -> dict:
    """
    Returns:
    {
        "predicted_class": 0 or 1,
        "label": "...",
        "probability": float | None
    }
    """
    try:
        logger.info("Starting prediction.")
        model, scaler, feature_columns = load_artifacts()
        input_df = prepare_input_data(user_input, feature_columns)
        input_scaled = scaler.transform(input_df)

        predicted_class = int(model.predict(input_scaled)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_scaled)[0][1])

        result = {
            "predicted_class": predicted_class,
            "label": (
                "High Chance of Admission"
                if predicted_class == 1
                else "Low Chance of Admission"
            ),
            "probability": probability,
        }

        logger.info("Prediction successful: %s", result)
        return result

    except ProjectException:
        raise
    except Exception as exc:
        logger.exception("Unexpected prediction error.")
        raise ProjectException("Failed to predict admission status.") from exc