from src.data_loader import load_data
from src.logger import get_logger
from src.preprocess import preprocess_data
from src.train import evaluate_model, save_artifacts, scale_data, train_model

logger = get_logger(__name__)


def run_training_pipeline():
    logger.info("Training pipeline started")

    df = load_data()
    x_train, x_test, y_train, y_test, feature_columns = preprocess_data(df)

    x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)
    model = train_model(x_train_scaled, y_train)
    metrics = evaluate_model(model, x_train_scaled, y_train, x_test_scaled, y_test)

    save_artifacts(model, scaler, feature_columns, metrics)

    logger.info("Training pipeline completed successfully")
    print("Training completed successfully.")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_training_pipeline()