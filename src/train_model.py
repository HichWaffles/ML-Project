from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import logger


def train_models():
    """Trains Random Forest, Logistic Regression, and Gradient Boosting classifiers
    on the preprocessed training data and saves each model."""

    data_dir = project_root / "data" / "train_test"

    logger.info("Loading training data...")
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")

    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")

    # Ensure targets are 1D arrays
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    trained_models = {}

    # -------------------------------------------------------------------------
    # Model 1: Random Forest
    # -------------------------------------------------------------------------
    logger.info("--- MODEL 1: Random Forest ---")
    logger.info("Setting up GridSearchCV for Random Forest Classifier...")

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")

    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    rf_grid = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )

    logger.info("Starting Grid Search...")
    rf_grid.fit(X_train, y_train)
    logger.info(f"Best parameters: {rf_grid.best_params_}")
    logger.info(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

    rf_model = rf_grid.best_estimator_
    _evaluate(rf_model, X_test, y_test)

    rf_path = models_dir / "churn_rf_model.joblib"
    joblib.dump(rf_model, rf_path)
    logger.info(f"Saved Random Forest model → {rf_path}")
    trained_models["random_forest"] = rf_model

    # -------------------------------------------------------------------------
    # Model 2: Logistic Regression
    # -------------------------------------------------------------------------
    logger.info("--- MODEL 2: Logistic Regression ---")
    logger.info("Setting up GridSearchCV for Logistic Regression...")

    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    lr_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "saga"],
        "penalty": ["l2"],
    }

    lr_grid = GridSearchCV(
        estimator=lr,
        param_grid=lr_param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )

    logger.info("Starting Grid Search...")
    lr_grid.fit(X_train, y_train)
    logger.info(f"Best parameters: {lr_grid.best_params_}")
    logger.info(f"Best CV ROC-AUC: {lr_grid.best_score_:.4f}")

    lr_model = lr_grid.best_estimator_
    _evaluate(lr_model, X_test, y_test)

    lr_path = models_dir / "churn_lr_model.joblib"
    joblib.dump(lr_model, lr_path)
    logger.info(f"Saved Logistic Regression model → {lr_path}")
    trained_models["logistic_regression"] = lr_model

    # -------------------------------------------------------------------------
    # Model 3: Gradient Boosting
    # -------------------------------------------------------------------------
    logger.info("--- MODEL 3: Gradient Boosting ---")
    logger.info("Setting up GridSearchCV for Gradient Boosting Classifier...")

    gb = GradientBoostingClassifier(random_state=42)

    gb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    gb_grid = GridSearchCV(
        estimator=gb,
        param_grid=gb_param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )

    logger.info("Starting Grid Search...")
    gb_grid.fit(X_train, y_train)
    logger.info(f"Best parameters: {gb_grid.best_params_}")
    logger.info(f"Best CV ROC-AUC: {gb_grid.best_score_:.4f}")

    gb_model = gb_grid.best_estimator_
    _evaluate(gb_model, X_test, y_test)

    gb_path = models_dir / "churn_gb_model.joblib"
    joblib.dump(gb_model, gb_path)
    logger.info(f"Saved Gradient Boosting model → {gb_path}")
    trained_models["gradient_boosting"] = gb_model

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("--- ALL MODELS TRAINED AND SAVED ---")
    for name, path in [
        ("Random Forest", "churn_rf_model.joblib"),
        ("Logistic Regression", "churn_lr_model.joblib"),
        ("Gradient Boosting", "churn_gb_model.joblib"),
    ]:
        logger.info(f"  {name:25s} → models/{path}")

    return trained_models


def _evaluate(model, X_test, y_test):
    """Logs classification report and ROC-AUC for a fitted model."""
    logger.info("Evaluation on Test Set:")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"ROC-AUC Score: {auc:.4f}")


if __name__ == "__main__":
    train_models()
