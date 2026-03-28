from pathlib import Path

import pandas as pd


def load_train_test_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads train/test split CSV files and returns X/y datasets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test
