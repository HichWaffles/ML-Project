from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def train_model():
    """Trains a Random Forest classifier on the preprocessed training data and saves it."""
    # 1. Load Data
    data_dir = project_root / "data" / "train_test"
    
    print("Loading training data...")
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")
    
    # Ensure targets are 1D arrays
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # 2. Initialize and Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 3. Quick Evaluation on Test Set
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.4f}")

    # 4. Save Model
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "churn_rf_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to {model_path}")

    return model

if __name__ == "__main__":
    train_model()
