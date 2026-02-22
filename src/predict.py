from pathlib import Path
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import logger

MODEL_PATHS = {
    "random_forest": project_root / "models" / "churn_rf_model.joblib",
    "logistic_regression": project_root / "models" / "churn_lr_model.joblib",
    "gradient_boosting": project_root / "models" / "churn_gb_model.joblib",
}


def load_model(model_path: Path):
    """Loads a trained machine learning model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_models.py first."
        )
    return joblib.load(model_path)


def predict(
    new_data: pd.DataFrame,
    model_name: str = "random_forest",
) -> pd.DataFrame:
    """
    Takes raw new data and predicts churn using the specified model.

    Parameters:
        - new_data:    pd.DataFrame containing the same features as the training data
        - model_name:  One of 'random_forest', 'logistic_regression', 'gradient_boosting'

    Returns:
        - pd.DataFrame with original features plus 'Churn_Prediction' and 'Churn_Probability'
    """
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_PATHS.keys())}"
        )

    model = load_model(MODEL_PATHS[model_name])

    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]

    results = new_data.copy()
    results["Churn_Prediction"] = predictions
    results["Churn_Probability"] = probabilities

    return results


def predict_all(new_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Runs predictions from all 3 models and returns a dict of results DataFrames,
    keyed by model name.

    Parameters:
        - new_data: pd.DataFrame containing the same features as the training data

    Returns:
        - dict mapping model name → DataFrame with 'Churn_Prediction' and 'Churn_Probability'
    """
    return {name: predict(new_data, model_name=name) for name in MODEL_PATHS}


def generate_reports(all_results: dict[str, pd.DataFrame], y_test: pd.Series):
    """Generates confusion matrices and ROC curves for all models."""
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Evaluation Reports...")

    # 1. Confusion Matrices
    for model_name, results in all_results.items():
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, results["Churn_Prediction"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = reports_dir / f"{model_name}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path)
        logger.info(f"Saved Confusion Matrix to {cm_path}")
        plt.close()

    # 2. ROC Curves (Combined)
    plt.figure(figsize=(10, 8))
    for model_name, results in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, results["Churn_Probability"])
        auc = roc_auc_score(y_test, results["Churn_Probability"])
        plt.plot(
            fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {auc:.4f})"
        )

    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves - All Models")
    plt.legend(loc="lower right")

    roc_path = reports_dir / "roc_curves_combined.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    logger.info(f"Saved Combined ROC Curves to {roc_path}")
    plt.close()


def main():
    """Demonstrates predictions from all 3 models using the saved test set."""
    logger.info("Loading test dataset for predictions...")
    data_dir = project_root / "data" / "train_test"
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    all_results = predict_all(X_test)

    for model_name, results in all_results.items():
        accuracy = (results["Churn_Prediction"] == y_test).mean()

        logger.info(f"--- Model: {model_name.replace('_', ' ').title()} ---")
        logger.info(
            f"Sample Predictions:\n{results[['Churn_Prediction', 'Churn_Probability']].head(10).to_string()}"
        )
        logger.info(f"Prediction Accuracy on Test Set: {accuracy:.4f}")

    generate_reports(all_results, y_test)


if __name__ == "__main__":
    main()
