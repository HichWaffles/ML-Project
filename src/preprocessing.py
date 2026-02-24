from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    apply_standard_scaler,
    extract_ip_features,
    filter_outliers,
    identify_redundant_features,
    identify_non_contributory_features,
    impute_missing_knn,
    remove_outliers_isolation_forest,
    split_columns_by_nan_threshold,
    target_encode,
    logger,
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
    "SupportTicketsCount": [-1, 999],
    "SatisfactionScore": [-1, 0, 99],
    "GeoIP": ["Unspecified", "Unknown"],
}

outlier_percentages = {"SupportTicketsCount": 0.05, "SatisfactionScore": 0.05}


def apply_ordinal_encoding(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features based on existing transaction and customer data in-place.
    """

    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = np.where(
            df["Frequency"] > 0, df["MonetaryTotal"] / df["Frequency"], 0
        )

    if "Recency" in df.columns and "CustomerTenure" in df.columns:
        df["TenureRatio"] = np.where(
            df["CustomerTenure"] > 0, df["Recency"] / df["CustomerTenure"], 0
        )

    if "SupportTicketsCount" in df.columns and "CustomerTenure" in df.columns:
        df["TicketIntensity"] = df["SupportTicketsCount"] / (df["CustomerTenure"] + 1)

    if "CancelledTrans" in df.columns and "Frequency" in df.columns:
        df["CancellationRate"] = np.where(
            df["Frequency"] > 0, df["CancelledTrans"] / df["Frequency"], 0
        )

    if "ZeroPriceCount" in df.columns and "TotalTrans" in df.columns:
        df["ZeroPriceRatio"] = np.where(
            df["TotalTrans"] > 0, df["ZeroPriceCount"] / df["TotalTrans"], 0
        )

    return df


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


def prune_nonessential_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    logger.info(f"Dropping {len(columns)} non-essential features.")
    return df.drop(columns=list(set(columns)), errors="ignore")


def prepare_features(df: pd.DataFrame, target_encoding=True) -> pd.DataFrame:
    df = apply_ordinal_encoding(df, ordinal_mappings)
    df = apply_one_hot_encoding(df, one_hot_cols)
    df = parse_ip(df)
    df = parse_registration_date(df)
    df = engineer_features(df)

    if target_encoding:
        df, _ = target_encode(df, "Country", smoothing=30)
        df, _ = target_encode(df, "GeoIP", smoothing=20)

    df = values_to_nan(df, columns_with_nan_values)

    df = filter_outliers(df, outlier_percentages)

    _, high_nan_cols = split_columns_by_nan_threshold(df, threshold=0.5)
    df = prune_nonessential_features(df, columns_to_drop + high_nan_cols)

    return df


def preprocess_data(df: pd.DataFrame, target_encoding=True) -> pd.DataFrame:
    df = prepare_features(df, target_encoding=target_encoding)

    redu_cols = identify_redundant_features(df, use_vif=False)
    irr_cols = identify_non_contributory_features(
        df, target_cols=["Churn"], threshold=0.01
    )

    df = prune_nonessential_features(df, redu_cols + irr_cols)

    df = filter_outliers(df, outlier_percentages)

    low_nan_cols, _ = split_columns_by_nan_threshold(df, threshold=0.5)
    df, _, _ = impute_missing_knn(df, target_columns=low_nan_cols)

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
    X_train = prune_nonessential_features(X_train, high_nan_cols)

    X_train, fitted_imputer, fitted_knn_scaler = impute_missing_knn(
        X_train, target_columns=low_nan_cols
    )
    X_train, fitted_final_scaler = apply_standard_scaler(X_train, target_col="Churn")

    redu_cols = identify_redundant_features(
        X_train, target_cols=["Churn"], use_vif=False
    )
    irr_cols = identify_non_contributory_features(
        X_train, target_cols=["Churn"], threshold=0.01
    )
    features_to_drop = redu_cols + irr_cols

    X_train = prune_nonessential_features(X_train, features_to_drop)

    y_train_clean = X_train["Churn"]
    X_train_clean = X_train.drop(columns=["Churn"], errors="ignore")

    pca = PCA(n_components=13, random_state=42)
    X_train_pca_array = pca.fit_transform(X_train_clean)
    logger.info("Remaining Variance after PCA: %s", pca.explained_variance_ratio_.sum())

    pca_cols = [f"PC{i+1}" for i in range(X_train_pca_array.shape[1])]
    X_train_pca = pd.DataFrame(
        X_train_pca_array, columns=pca_cols, index=X_train_clean.index
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_pca, y_train_clean)

    fitted_artifacts = {
        "country_enc": country_enc,
        "geoip_enc": geoip_enc,
        "high_nan_cols": high_nan_cols,
        "low_nan_cols": low_nan_cols,
        "imputer": fitted_imputer,
        "knn_scaler": fitted_knn_scaler,
        "final_scaler": fitted_final_scaler,
        "features_to_drop": features_to_drop,
        "pca": pca,
    }

    return X_train_bal, y_train_bal, fitted_artifacts


def transform_test(X_test: pd.DataFrame, fitted_artifacts: dict) -> pd.DataFrame:
    X_test = X_test.copy()

    X_test, _ = target_encode(
        X_test, "Country", encoder=fitted_artifacts["country_enc"]
    )
    X_test, _ = target_encode(X_test, "GeoIP", encoder=fitted_artifacts["geoip_enc"])

    X_test = prune_nonessential_features(X_test, fitted_artifacts["high_nan_cols"])
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
    X_test = prune_nonessential_features(X_test, fitted_artifacts["features_to_drop"])
    pca = fitted_artifacts["pca"]
    X_test_pca_array = pca.transform(X_test)

    pca_cols = [f"PC{i+1}" for i in range(X_test_pca_array.shape[1])]
    X_test_pca = pd.DataFrame(X_test_pca_array, columns=pca_cols, index=X_test.index)
    return X_test_pca


def save_splits(X_train, X_test, y_train, y_test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)
    logger.info(f"Saved processed splits to {out_dir}")


def save_processed_data(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "processed_data.csv", index=False)
    logger.info(f"Saved processed data to {out_dir}")


def main():
    data_path = (
        project_root / "data" / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"
    )
    df = pd.read_csv(data_path)

    df = prepare_features(df, target_encoding=False)
    save_processed_data(df, out_dir=project_root / "data" / "processed")

    X_train, X_test, y_train, y_test = split_data(df)

    logger.info("Fitting transformations on X_train...")
    X_train, y_train, fitted_artifacts = fit_transform_train(X_train, y_train)

    # Save  the fitted artifacts for use in test transformation and future predictions in model_dir
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_artifacts, model_dir / "fitted_artifacts.joblib")
    logger.info(f"Saved fitted artifacts → {model_dir / 'fitted_artifacts.joblib'}")

    logger.info("Applying transformations to X_test...")
    X_test = transform_test(X_test, fitted_artifacts)

    save_splits(
        X_train, X_test, y_train, y_test, out_dir=project_root / "data" / "train_test"
    )


if __name__ == "__main__":
    main()
