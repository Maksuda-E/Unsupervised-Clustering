import pickle

import pandas as pd

from src.config import FEATURE_COLUMNS_FILE_PATH, MODEL_FILE_PATH, SCALER_FILE_PATH
from src.custom_exception import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_prediction_artifacts():
    try:
        logger.info("Loading prediction artifacts")
        model = _load_pickle(MODEL_FILE_PATH)
        scaler = _load_pickle(SCALER_FILE_PATH)
        feature_columns = _load_pickle(FEATURE_COLUMNS_FILE_PATH)
        logger.info("Prediction artifacts loaded successfully")
        return model, scaler, feature_columns
    except FileNotFoundError as exc:
        logger.exception("One or more artifact files are missing")
        raise ProjectException(
            "Artifacts not found. Run training first to generate model/scaler/features."
        ) from exc
    except Exception as exc:
        logger.exception("Error while loading prediction artifacts")
        raise ProjectException("Failed to load prediction artifacts.") from exc


def prepare_input_data(user_input: dict, feature_columns: list[str]) -> pd.DataFrame:
    """
    Match training preprocessing:
    - raw columns
    - cast University_Rating and Research as categorical
    - pd.get_dummies
    - align to training feature columns
    """
    try:
        logger.info("Preparing input data for prediction")

        required_keys = {
            "GRE_Score",
            "TOEFL_Score",
            "University_Rating",
            "SOP",
            "LOR",
            "CGPA",
            "Research",
        }

        missing = required_keys.difference(user_input.keys())
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

        logger.info("Input data prepared successfully")
        return input_df

    except ProjectException:
        logger.exception("ProjectException while preparing input")
        raise
    except Exception as exc:
        logger.exception("Unexpected error while preparing input")
        raise ProjectException("Failed to prepare input data.") from exc


def predict_admission(user_input: dict) -> dict:
    """
    Returns both class and probability so app.py can show a true percentage.
    """
    try:
        model, scaler, feature_columns = load_prediction_artifacts()
        prepared_input = prepare_input_data(user_input, feature_columns)
        scaled_input = scaler.transform(prepared_input)

        predicted_class = int(model.predict(scaled_input)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(scaled_input)[0][1])

        result = {
            "predicted_class": predicted_class,
            "label": (
                "High Chance of Admission"
                if predicted_class == 1
                else "Low Chance of Admission"
            ),
            "probability": probability,
        }

        logger.info("Prediction completed successfully: %s", result)
        return result

    except ProjectException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise ProjectException("Prediction failed.") from exc