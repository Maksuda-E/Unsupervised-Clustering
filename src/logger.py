# Import the built in logging module.
import logging

# Import os so we can create the logs folder.
import os


# Create the logs folder if it does not exist.
os.makedirs("logs", exist_ok=True)

# Store the log file path.
LOG_FILE_PATH = os.path.join("logs", "project.log")

# Configure the logging system.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Create a logger object that other files can import.
logger = logging.getLogger("unsupervised_clustering")