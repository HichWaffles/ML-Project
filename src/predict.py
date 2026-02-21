from pathlib import Path
import sys
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import prepare_features, transform_test, drop_unnecessary_columns

MODEL_PATH = project_root / "models" / "churn_rf_model.joblib"


def load_model(model_path=MODEL_PATH):
    """Loads the trained machine learning model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_model.py first."
        )
    return joblib.load(model_path)


def predict(new_data: pd.DataFrame, model_path=MODEL_PATH) -> pd.DataFrame:
    """
    Takes raw new data, applies static preprocessing, then dynamic transformations
    using fitted artifacts before predicting churn probabilities.
    Parameters:
        - new_data: pd.DataFrame containing the same features as the training data (except target)
        - model_path: Path to the saved model file
    Returns:
        - pd.DataFrame with original features plus 'Churn_Prediction' and 'Churn_Probability' columns
    """
    model = load_model(model_path)

    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]

    results = new_data.copy()
    results["Churn_Prediction"] = predictions
    results["Churn_Probability"] = probabilities

    return results


def main():
    """Main function to demonstrate predictions using the saved test set."""
    print("Loading test dataset...")
    data_dir = project_root / "data" / "train_test"
    X_test = pd.read_csv(data_dir / "X_test.csv")

    print("Running predictions...")
    results = predict(X_test)

    # Display the first 10 predictions
    print("\n--- Example Predictions ---")
    print(results[["Churn_Prediction", "Churn_Probability"]].head(10))
    # Show accuracy of predictions if we have true labels
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    accuracy = (results["Churn_Prediction"] == y_test).mean()
    print(f"\nPrediction Accuracy on Test Set: {accuracy:.4f}")


if __name__ == "__main__":
    main()
