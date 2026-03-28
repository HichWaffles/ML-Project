from pathlib import Path

import pandas as pd

from src.utils.paths import RAW_DATA_PATH


def load_train_test_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train/test split CSV files and return X/y datasets."""
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)


def save_processed_data(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "processed_data.csv", index=False)


def load_raw_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the raw customer dataset used by preprocessing."""
    if data_path is None:
        data_path = RAW_DATA_PATH
    return pd.read_csv(data_path)
