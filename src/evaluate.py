# Import pandas for creating result tables.
import pandas as pd

# Import KMeans for clustering.
from sklearn.cluster import KMeans

# Import silhouette_score for cluster evaluation.
from sklearn.metrics import silhouette_score

# Import custom exception handling.
from src.custom_exception import ProjectException

# Import logger.
from src.logger import logger


# Create a class to calculate clustering evaluation scores.
class ClusterEvaluator:
    # Define the constructor.
    def __init__(self, random_state: int, init: str, n_init: int, max_iter: int):
        # Save random state.
        self.random_state = random_state

        # Save initialization method.
        self.init = init

        # Save n_init value.
        self.n_init = n_init

        # Save max iterations.
        self.max_iter = max_iter

    # Create a method to calculate elbow scores.
    def calculate_elbow(self, X, k_range) -> pd.DataFrame:
        # Start a try block.
        try:
            # Create an empty list for result rows.
            rows = []

            # Loop through each cluster number.
            for k in k_range:
                # Create the KMeans model.
                model = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    init=self.init,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                )

                # Fit the model on the input data.
                model.fit(X)

                # Save the inertia value for this k.
                rows.append({"cluster": k, "WCSS_Score": round(model.inertia_, 6)})

            # Convert the result list into a dataframe.
            result = pd.DataFrame(rows)

            # Log completion.
            logger.info("Elbow scores calculated successfully")

            # Return the dataframe.
            return result

        # Catch evaluation errors.
        except Exception as exc:
            # Log the full error.
            logger.exception("Elbow calculation failed")

            # Raise a project specific exception.
            raise ProjectException(f"Error calculating elbow scores: {exc}") from exc

    # Create a method to calculate silhouette scores.
    def calculate_silhouette(self, X, k_range) -> pd.DataFrame:
        # Start a try block.
        try:
            # Create an empty list for result rows.
            rows = []

            # Loop through each cluster number.
            for k in k_range:
                # Create the KMeans model.
                model = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    init=self.init,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                )

                # Fit the model and get cluster labels.
                labels = model.fit_predict(X)

                # Calculate silhouette score.
                score = silhouette_score(X, labels)

                # Save the result row.
                rows.append({"cluster": k, "Silhouette_Score": round(score, 6)})

            # Convert the results into a dataframe.
            result = pd.DataFrame(rows)

            # Log completion.
            logger.info("Silhouette scores calculated successfully")

            # Return the dataframe.
            return result

        # Catch evaluation errors.
        except Exception as exc:
            # Log the full error.
            logger.exception("Silhouette calculation failed")

            # Raise a project specific exception.
            raise ProjectException(f"Error calculating silhouette scores: {exc}") from exc