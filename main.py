# Import the training pipeline function.
from src.train import run_training_pipeline


# Run the training pipeline only when the script is executed directly.
if __name__ == "__main__":
    # Start model training and artifact generation.
    run_training_pipeline()