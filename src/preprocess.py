# This line imports pandas for data processing
import pandas as pd

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function cleans the dataset
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # This line starts the try block
    try:
        # This line logs the start of data cleaning
        logger.info("Starting data cleaning")

        # This line creates a copy of the dataset
        df = df.copy()

        # This line removes leading and trailing spaces from column names
        df.columns = df.columns.str.strip()

        # This line removes duplicate rows
        df = df.drop_duplicates()

        # This line removes CustomerID if it exists because it is not useful for clustering
        if "CustomerID" in df.columns:
            df = df.drop("CustomerID", axis=1)

        # This line converts Gender into numeric values if the column exists
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

        # This line fills missing values in numeric columns with the median
        for column in df.select_dtypes(include=["number"]).columns:
            if df[column].isnull().sum() > 0:
                df[column] = df[column].fillna(df[column].median())

        # This line logs that data cleaning completed successfully
        logger.info("Data cleaning completed successfully")

        # This line returns the cleaned dataset
        return df

    # This block handles cleaning errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during data cleaning")

        # This line raises a custom exception
        raise ProjectException(f"Failed to clean data: {exc}")

# This function selects the clustering features
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    # This line starts the try block
    try:
        # This line logs feature selection
        logger.info("Selecting clustering features")

        # This line selects the three main features used in the notebook
        x = df[["Age", "Annual_Income", "Spending_Score"]]

        # This line returns the selected features
        return x

    # This block handles feature selection errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while selecting features")

        # This line raises a custom exception
        raise ProjectException(f"Failed to select features: {exc}")