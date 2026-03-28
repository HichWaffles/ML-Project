from pathlib import Path

import joblib

from src.utils.paths import MODELS_DIR


MODEL_PATHS = {
    "random_forest": MODELS_DIR / "churn_rf_model.joblib",
    "logistic_regression": MODELS_DIR / "churn_lr_model.joblib",
    "gradient_boosting": MODELS_DIR / "churn_gb_model.joblib",
    "xgboost": MODELS_DIR / "churn_xgb_model.joblib",
}


def get_model_path(model_name: str) -> Path:
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_PATHS.keys())}"
        )
    return MODEL_PATHS[model_name]


def load_model(model_path: Path):
    """Load a trained machine learning model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_models.py first."
        )
    return joblib.load(model_path)
