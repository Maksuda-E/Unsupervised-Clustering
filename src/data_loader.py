# Import pandas for CSV loading.
import pandas as pd

# Import the custom exception class.
from src.custom_exception import ProjectException

# Import the logger object.
from src.logger import logger


# Create a class to manage dataset loading.
class DataLoader:
    # Define the constructor.
    def __init__(self, data_path: str):
        # Save the file path in the object.
        self.data_path = data_path

    # Create a method to load the dataset.
    def load_data(self) -> pd.DataFrame:
        # Start a try block for safe loading.
        try:
            # Log the dataset loading start.
            logger.info("Loading dataset from %s", self.data_path)

            # Read the CSV file.
            df = pd.read_csv(self.data_path)

            # Log the loaded dataset shape.
            logger.info("Dataset loaded with shape %s", df.shape)

            # Return the dataframe.
            return df

        # Catch any loading error.
        except Exception as exc:
            # Write the full exception to the log.
            logger.exception("Failed to load dataset")

            # Raise a cleaner project specific exception.
            raise ProjectException(f"Error loading data: {exc}") from exc