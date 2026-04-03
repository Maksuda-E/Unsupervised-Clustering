import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, THRESHOLD
from src.custom_exception import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(df: pd.DataFrame):
    """
    Notebook-faithful preprocessing:
    1. Convert Admit_Chance into binary target using threshold 0.80
    2. Drop Serial_No
    3. Cast University_Rating and Research to object
    4. One-hot encode those columns
    5. Split train/test with stratify=y
    """
    try:
        logger.info("Starting preprocessing.")

        data = df.copy()
        data.columns = [col.strip() for col in data.columns]

        required_columns = {
            "Serial_No",
            "GRE_Score",
            "TOEFL_Score",
            "University_Rating",
            "SOP",
            "LOR",
            "CGPA",
            "Research",
            "Admit_Chance",
        }

        missing_cols = required_columns - set(data.columns)
        if missing_cols:
            raise ProjectException(f"Missing required columns: {sorted(missing_cols)}")

        data["Admit_Chance"] = (data["Admit_Chance"] >= THRESHOLD).astype(int)

        data = data.drop(columns=["Serial_No"])

        data["University_Rating"] = data["University_Rating"].astype("object")
        data["Research"] = data["Research"].astype("object")

        clean_data = pd.get_dummies(
            data,
            columns=["University_Rating", "Research"],
            dtype=int,
        )

        x = clean_data.drop(columns=["Admit_Chance"])
        y = clean_data["Admit_Chance"]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        logger.info(
            "Preprocessing completed. x_train=%s, x_test=%s",
            x_train.shape,
            x_test.shape,
        )

        return x_train, x_test, y_train, y_test, list(x_train.columns)

    except ProjectException:
        logger.exception("ProjectException occurred during preprocessing.")
        raise
    except Exception as exc:
        logger.exception("Unexpected error during preprocessing.")
        raise ProjectException("Failed to preprocess data.") from exc