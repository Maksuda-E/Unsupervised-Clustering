import pandas as pd

from src.config import DATA_FILE_PATH
from src.custom_exception import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path=DATA_FILE_PATH) -> pd.DataFrame:
    try:
        logger.info("Loading dataset from %s", file_path)
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully with shape %s", df.shape)
        return df
    except FileNotFoundError as exc:
        logger.exception("Dataset file not found")
        raise ProjectException(
            f"Dataset file not found at: {file_path}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading dataset")
        raise ProjectException("Failed to load dataset.") from exc