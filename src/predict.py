# Import pandas for building single row input data.
import pandas as pd

# Import artifact and model configuration.
from src.config import ArtifactConfig, ModelConfig

# Import custom exception handling.
from src.custom_exception import ProjectException

# Import logger.
from src.logger import logger

# Import the object loader utility.
from src.utils import load_object


# Create a predictor class for app use.
class ClusterPredictor:
    # Define the constructor.
    def __init__(self):
        # Start a try block.
        try:
            # Load the trained KMeans model.
            self.model = load_object(ArtifactConfig.MODEL_PATH)

            # Load the fitted scaler.
            self.scaler = load_object(ArtifactConfig.SCALER_PATH)

            # Log success.
            logger.info("Prediction artifacts loaded successfully")

        # Catch artifact loading errors.
        except Exception as exc:
            # Log the full exception.
            logger.exception("Failed to load prediction artifacts")

            # Raise a clean project error.
            raise ProjectException(
                "Could not load model artifacts. Run python main.py first."
            ) from exc

    # Create a method to prepare input data.
    def _prepare_input(
        self,
        age: int,
        annual_income: int,
        spending_score: int,
        gender: str | None = None,
    ) -> pd.DataFrame:
        # Build a one row dataframe with model feature names.
        input_df = pd.DataFrame(
            {
                "Age": [age],
                "Annual_Income": [annual_income],
                "Spending_Score": [spending_score],
            }
        )

        # Return the prepared dataframe.
        return input_df

    # Create a method to predict a single customer cluster.
    def predict_single(
        self,
        age: int,
        annual_income: int,
        spending_score: int,
        gender: str | None = None,
    ) -> int:
        # Start a try block.
        try:
            # Prepare the input dataframe.
            input_df = self._prepare_input(
                age=age,
                annual_income=annual_income,
                spending_score=spending_score,
                gender=gender,
            )

            # Scale the input using the saved scaler.
            scaled_input = self.scaler.transform(input_df[ModelConfig.FEATURE_COLUMNS])

            # Predict the cluster id.
            prediction = self.model.predict(scaled_input)[0]

            # Log prediction success.
            logger.info("Prediction completed successfully")

            # Return the prediction as an integer.
            return int(prediction)

        # Catch prediction errors.
        except Exception as exc:
            # Log the full exception.
            logger.exception("Prediction failed")

            # Raise a project specific exception.
            raise ProjectException(f"Prediction failed: {exc}") from exc