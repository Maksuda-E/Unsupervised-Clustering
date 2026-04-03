# Import json for saving dictionaries as JSON files.
import json

# Import os for folder creation.
import os

# Import Any for flexible type hints.
from typing import Any

# Import joblib for saving and loading Python objects.
import joblib


# Create a function to make sure folders exist.
def ensure_directories(*paths: str) -> None:
    # Loop through every given folder path.
    for path in paths:
        # Create the folder if it does not exist.
        os.makedirs(path, exist_ok=True)


# Create a function to save Python objects.
def save_object(file_path: str, obj: Any) -> None:
    # Get the parent folder path from the file path.
    parent_dir = os.path.dirname(file_path)

    # Check whether a parent folder exists in the path.
    if parent_dir:
        # Create the folder if needed.
        os.makedirs(parent_dir, exist_ok=True)

    # Save the object using joblib.
    joblib.dump(obj, file_path)


# Create a function to load Python objects.
def load_object(file_path: str) -> Any:
    # Load and return the object from disk.
    return joblib.load(file_path)


# Create a function to save a dictionary as JSON.
def save_json(file_path: str, data: dict) -> None:
    # Get the parent folder path.
    parent_dir = os.path.dirname(file_path)

    # Create the parent folder if it exists in the path.
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Open the file in write mode with UTF 8 encoding.
    with open(file_path, "w", encoding="utf-8") as file:
        # Write the JSON data with indentation for readability.
        json.dump(data, file, indent=4, ensure_ascii=False)