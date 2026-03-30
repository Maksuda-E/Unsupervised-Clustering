# This line imports pandas for reading the dataset
import pandas as pd

# This line imports the logger function
from src.logger import get_logger

# This line imports the custom project exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function loads the CSV dataset
def load_data(file_path: str) -> pd.DataFrame:
    # This line starts a try block for safe execution
    try:
        # This line logs that data loading has started
        logger.info("Starting data loading from file")

        # This line reads the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # This line logs that data loading finished successfully
        logger.info("Data loaded successfully")

        # This line returns the loaded DataFrame
        return df

    # This block handles any exception during file loading
    except Exception as exc:
        # This line logs the error message
        logger.error("Error occurred while loading data")

        # This line raises a custom project exception
        raise ProjectException(f"Failed to load data: {exc}")