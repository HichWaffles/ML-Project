from pathlib import Path
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    apply_standard_scaler,
    extract_ip_features,
    get_features_to_destroy,
    get_irrelevant_features,
    impute_missing_knn,
    split_columns_by_nan_threshold,
    target_encode,
)

ordinal_mappings = {
    "RFMSegment": {"Dormants": 1, "Potentiels": 2, "Fidèles": 3, "Champions": 4},
    "AgeCategory": {
        "Inconnu": np.nan,
        "18-24": 1,
        "25-34": 2,
        "35-44": 3,
        "45-54": 4,
        "55-64": 5,
        "65+": 6,
    },
    "SpendingCategory": {"Low": 1, "Medium": 2, "High": 3, "VIP": 4},
    "PreferredTimeOfDay": {"Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4},
    "LoyaltyLevel": {"Nouveau": 1, "Jeune": 2, "Établi": 3, "Ancien": 4},
    "ChurnRiskCategory": {"Faible": 1, "Moyen": 2, "Élevé": 3, "Critique": 4},
    "BasketSizeCategory": {"Petit": 1, "Moyen": 2, "Grand": 3},
}

one_hot_cols = [
    "CustomerType",
    "FavoriteSeason",
    "Region",
    "WeekendPreference",
    "ProductDiversity",
    "Gender",
    "AccountStatus",
]

columns_to_drop = [
    "CustomerID",
    "NewsletterSubscribed",
    "LastLoginIP",
    "RegistrationDate",
]

columns_with_nan_values = {
    "SupportTickets": [-1, 999],
    "SatisfactionScore": [-1, 0, 99],
    "GeoIP": ["Unspecified", "Unknown"],
}


def apply_ordinal_encoding(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def parse_ip(df: pd.DataFrame) -> pd.DataFrame:
    df[["GeoIP"]] = df["LastLoginIP"].apply(extract_ip_features)
    return df


def parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"], format="mixed", dayfirst=True, errors="coerce"
    )
    df["RegistrationYear"] = df["RegistrationDate"].dt.year
    df["RegistrationMonth"] = df["RegistrationDate"].dt.month
    df["RegistrationDay"] = df["RegistrationDate"].dt.day
    df["RegistrationDayOfWeek"] = df["RegistrationDate"].dt.dayofweek

    max_date = df["RegistrationDate"].max()
    df["DaysSinceRegistration"] = (max_date - df["RegistrationDate"]).dt.days

    return df


def values_to_nan(df: pd.DataFrame, columns_with_nan_values: dict) -> pd.DataFrame:
    for col, values in columns_with_nan_values.items():
        if col in df.columns:
            df[col] = df[col].replace(values, np.nan)
    return df


def drop_unnecessary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=list(set(columns)), errors="ignore")


def prepare_features(df: pd.DataFrame, target_encoding=True) -> pd.DataFrame:
    df = apply_ordinal_encoding(df, ordinal_mappings)
    df = apply_one_hot_encoding(df, one_hot_cols)
    df = parse_ip(df)
    df = parse_registration_date(df)

    if target_encoding:
        df, _ = target_encode(df, "Country", smoothing=30)
        df, _ = target_encode(df, "GeoIP", smoothing=20)

    df = values_to_nan(df, columns_with_nan_values)

    _, high_nan_cols = split_columns_by_nan_threshold(df, threshold=0.5)
    df = drop_unnecessary_columns(df, columns_to_drop + high_nan_cols)

    return df


def preprocess_data(df: pd.DataFrame, target_encoding=True) -> pd.DataFrame:
    df = prepare_features(df, target_encoding=target_encoding)

    redu_cols = get_features_to_destroy(df, use_vif=False)
    irr_cols = get_irrelevant_features(df, target_cols=["Churn"], threshold=0.01)

    df = drop_unnecessary_columns(df, redu_cols + irr_cols)

    return df


def split_data(df: pd.DataFrame, target_col: str = "Churn"):
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)


def fit_transform_train(X_train: pd.DataFrame, y_train: pd.Series):
    X_train = X_train.copy()
    X_train["Churn"] = y_train

    X_train, country_enc = target_encode(
        X_train, "Country", target_col="Churn", smoothing=30
    )
    X_train, geoip_enc = target_encode(
        X_train, "GeoIP", target_col="Churn", smoothing=20
    )

    low_nan_cols, high_nan_cols = split_columns_by_nan_threshold(X_train, threshold=0.5)
    X_train = drop_unnecessary_columns(X_train, high_nan_cols)

    X_train, fitted_imputer, fitted_knn_scaler = impute_missing_knn(
        X_train, target_columns=low_nan_cols
    )
    X_train, fitted_final_scaler = apply_standard_scaler(X_train, target_col="Churn")

    redu_cols = get_features_to_destroy(X_train, target_cols=["Churn"], use_vif=False)
    irr_cols = get_irrelevant_features(X_train, target_cols=["Churn"], threshold=0.01)
    features_to_drop = redu_cols + irr_cols
    
    X_train = drop_unnecessary_columns(X_train, features_to_drop)
    
    y_train_clean = X_train["Churn"]
    X_train_clean = X_train.drop(columns=["Churn"], errors="ignore")

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_clean, y_train_clean)

    fitted_artifacts = {
        "country_enc": country_enc,
        "geoip_enc": geoip_enc,
        "high_nan_cols": high_nan_cols,
        "low_nan_cols": low_nan_cols,
        "imputer": fitted_imputer,
        "knn_scaler": fitted_knn_scaler,
        "final_scaler": fitted_final_scaler,
        "features_to_drop": features_to_drop,
    }

    return X_train_bal, y_train_bal, fitted_artifacts


def transform_test(X_test: pd.DataFrame, fitted_artifacts: dict) -> pd.DataFrame:
    X_test = X_test.copy()

    X_test, _ = target_encode(
        X_test, "Country", encoder=fitted_artifacts["country_enc"]
    )
    X_test, _ = target_encode(X_test, "GeoIP", encoder=fitted_artifacts["geoip_enc"])

    X_test = drop_unnecessary_columns(X_test, fitted_artifacts["high_nan_cols"])
    X_test, _, _ = impute_missing_knn(
        X_test,
        target_columns=fitted_artifacts["low_nan_cols"],
        imputer=fitted_artifacts["imputer"],
        scaler=fitted_artifacts["knn_scaler"],
    )
    X_test, _ = apply_standard_scaler(
        X_test,
        scaler=fitted_artifacts["final_scaler"],
        target_col="Churn",
    )
    X_test = drop_unnecessary_columns(X_test, fitted_artifacts["features_to_drop"])

    return X_test


def save_splits(X_train, X_test, y_train, y_test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)
    print(f"Saved processed splits to {out_dir}")


def save_processed_data(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "processed_data.csv", index=False)
    print(f"Saved processed data to {out_dir}")


def main():
    data_path = (
        project_root / "data" / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"
    )
    df = pd.read_csv(data_path)

    df = prepare_features(df, target_encoding=False)
    save_processed_data(df, out_dir=project_root / "data" / "processed")

    X_train, X_test, y_train, y_test = split_data(df)

    print("Fitting transformations on X_train...")
    X_train, y_train, fitted_artifacts = fit_transform_train(X_train, y_train)

    print("Applying transformations to X_test...")
    X_test = transform_test(X_test, fitted_artifacts)

    save_splits(
        X_train, X_test, y_train, y_test, out_dir=project_root / "data" / "train_test"
    )


if __name__ == "__main__":
    main()
