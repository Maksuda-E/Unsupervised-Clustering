# This line imports the logging module for logging messages
import logging

# This line imports os for folder creation
import os

# This line imports the logs folder path and log file path from config
from src.config import LOGS_DIR, LOG_FILE_PATH

# This line creates the logs folder if it does not already exist
os.makedirs(LOGS_DIR, exist_ok=True)

# This line configures the logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

# This function returns a logger object for the given file name
def get_logger(name: str):
    # This line returns a logger with the given name
    return logging.getLogger(name)