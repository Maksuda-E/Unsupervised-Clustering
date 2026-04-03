from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

DATA_FILE_PATH = DATA_DIR / "Admission_Predict_Ver1.1.csv"

MODEL_FILE_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_FILE_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_FILE_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
METRICS_FILE_PATH = ARTIFACTS_DIR / "metrics.json"

RANDOM_STATE = 123
TEST_SIZE = 0.20
THRESHOLD = 0.80

APP_TITLE = "UCLA Admission Predictor"