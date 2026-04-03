from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

# Dataset
DATA_FILE_PATH = DATA_DIR / "Admission_Predict_Ver1.1.csv"

# Artifacts
MODEL_FILE_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_FILE_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_FILE_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
METRICS_FILE_PATH = ARTIFACTS_DIR / "metrics.json"

# Reproducibility
RANDOM_STATE = 123
TEST_SIZE = 0.20

# App metadata
APP_TITLE = "Graduate Admission Prediction"
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0
THRESHOLD = 0.80