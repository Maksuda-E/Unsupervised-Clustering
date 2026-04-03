# This line imports the os module so we can work with folder paths
import os

# This line gets the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)

# This line gets the src folder path
SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)

# This line gets the main project folder path
PROJECT_DIR = os.path.dirname(SRC_DIR)

# This line creates the full path for the data folder
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# This line creates the full path for the artifacts folder
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# This line creates the full path for the logs folder
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

# This line creates the full path for the dataset file
DATA_FILE_PATH = os.path.join(DATA_DIR, "mall_customers.csv")

# This line creates the full path for the saved clustering model file
MODEL_FILE_PATH = os.path.join(ARTIFACTS_DIR, "clustering_model.pkl")

# This line creates the full path for the saved scaler file
SCALER_FILE_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

# This line creates the full path for the saved metrics file
METRICS_FILE_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

# This line creates the full path for the log file
LOG_FILE_PATH = os.path.join(LOGS_DIR, "project.log")

# This line sets the random state for reproducibility
RANDOM_STATE = 42

# This line sets the number of clusters for the final model
N_CLUSTERS = 6
