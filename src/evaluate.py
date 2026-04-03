import json

from src.config import METRICS_FILE_PATH
from src.custom_exception import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)


def load_metrics(file_path=METRICS_FILE_PATH):
    try:
        logger.info("Loading metrics from %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        logger.info("Metrics loaded successfully.")
        return metrics
    except FileNotFoundError as exc:
        logger.exception("Metrics file not found.")
        raise ProjectException(f"Metrics file not found at: {file_path}") from exc
    except Exception as exc:
        logger.exception("Error while loading metrics.")
        raise ProjectException("Failed to load metrics.") from exc