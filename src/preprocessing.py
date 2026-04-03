# Import pandas for dataframe handling.
import pandas as pd

# Import StandardScaler for feature scaling.
from sklearn.preprocessing import StandardScaler

# Import the custom project exception.
from src.custom_exception import ProjectException

# Import the shared logger.
from src.logger import logger


# Create a preprocessing class.
class DataPreprocessor:
    # Save the feature columns and create the scaler.
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()

    # Check that all required columns exist.
    def validate_columns(self, df: pd.DataFrame) -> None:
        missing_columns = [col for col in self.feature_columns if col not in df.columns]
        if missing_columns:
            raise ProjectException(f"Missing required columns: {missing_columns}")

    # Fit the scaler and transform the training data.
    def fit_transform(self, df: pd.DataFrame):
        try:
            self.validate_columns(df)
            logger.info("Fitting scaler on columns: %s", self.feature_columns)
            scaled_array = self.scaler.fit_transform(df[self.feature_columns])
            logger.info("Scaling completed successfully")
            return scaled_array
        except Exception as exc:
            logger.exception("fit_transform preprocessing failed")
            raise ProjectException(f"Error during fit_transform: {exc}") from exc

    # Transform new data using the already fitted scaler.
    def transform(self, df: pd.DataFrame):
        try:
            self.validate_columns(df)
            scaled_array = self.scaler.transform(df[self.feature_columns])
            logger.info("Transform completed successfully")
            return scaled_array
        except Exception as exc:
            logger.exception("transform preprocessing failed")
            raise ProjectException(f"Error during transform: {exc}") from exc