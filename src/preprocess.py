import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, THRESHOLD
from src.custom_exception import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(df: pd.DataFrame):
    """
    Match the notebook flow:

    1. Convert Admit_Chance to binary: >= 0.8 -> 1 else 0
    2. Drop Serial_No
    3. Cast University_Rating and Research to categorical/object
    4. One-hot encode
    5. Split into train/test with stratify=y
    """
    try:
        logger.info("Preprocessing started")

        data = df.copy()

        # Normalize column names if CSV has leading/trailing spaces
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

        missing = required_columns.difference(data.columns)
        if missing:
            raise ProjectException(f"Missing required columns: {sorted(missing)}")

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
            "Preprocessing completed successfully. x_train=%s x_test=%s",
            x_train.shape,
            x_test.shape,
        )

        return x_train, x_test, y_train, y_test, list(x_train.columns)

    except ProjectException:
        logger.exception("ProjectException in preprocessing")
        raise
    except Exception as exc:
        logger.exception("Unexpected error during preprocessing")
        raise ProjectException("Failed to preprocess data.") from exc