from collections import defaultdict
from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import json

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.utils import (
    apply_standard_scaler,
    compute_days_since_registration,
    engineer_features,
    filter_outliers,
    identify_redundant_features,
    identify_non_contributory_features,
    impute_missing_knn,
    load_raw_data,
    logger,
    parse_ip,
    parse_registration_date,
    PROCESSED_DIR,
    MODELS_DIR,
    save_processed_data,
    save_splits,
    split_columns_by_nan_threshold,
    target_encode,
    clean_column_name,
    TRAIN_TEST_DIR,
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
    "Recency",  # Dropped because it basically already tells you everything about the target variable (churners have very high recency), and it doesn't make sense to keep it after creating TenureRatio. It also has a lot of missing values, and the few non-missing values are likely to be unreliable given the churn pattern.
    "ChurnRiskCategory",  # Dropped because it's a target leakage variable (it was created by the marketing team based on their assessment of how likely each customer is to churn, which is basically the same thing we're trying to predict). It also has a lot of missing values and is very imbalanced, so it would be hard to impute or use effectively even if we wanted to keep it.
    # NOTE: RegistrationDate is NOT dropped here; it must survive into
    # fit_transform_train / transform_test so compute_days_since_registration
    # can read it. It is dropped there, after DaysSinceRegistration is created.
]

columns_with_nan_values = {
    "SupportTicketsCount": [-1, 999],
    "SatisfactionScore": [-1, 0, 99],
    "GeoIP": ["Unspecified", "Unknown"],
    "Gender": ["Unknown"],
    "WeekendPreference": ["Inconnu"],
}

outlier_percentages = {"SupportTicketsCount": 0.05, "SatisfactionScore": 0.05}
TARGET_COL = "Churn"


def apply_ordinal_encoding(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def values_to_nan(df: pd.DataFrame, columns_with_nan_values: dict) -> pd.DataFrame:
    for col, values in columns_with_nan_values.items():
        if col in df.columns:
            df[col] = df[col].replace(values, np.nan)
    return df


def prune_nonessential_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    logger.info(f"Dropping {len(columns)} non-essential features. Examples: {columns}")
    return df.drop(columns=list(set(columns)), errors="ignore")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies purely structural transformations that are safe to run on the full
    dataset before splitting (no statistics derived from the target column,
    no outlier removal, no imputation).

    One-hot encoding, target encoding, outlier removal, imputation, and scaling are all deferred
    to fit_transform_train() / transform_test() to prevent data leakage.
    """
    # Parse IP first so GeoIP exists before placeholder-to-NaN replacement.
    df = parse_ip(df)
    df = values_to_nan(df, columns_with_nan_values)
    df = apply_ordinal_encoding(df, ordinal_mappings)
    df = parse_registration_date(df)
    df = engineer_features(df)

    df = prune_nonessential_features(df, columns_to_drop)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """EDA / exploratory helper only — NOT used in the training pipeline.

    WARNING: This function operates on the full (unsplit) DataFrame. Calling it
    before train_test_split will produce data leakage if the result is used for
    model training. Use fit_transform_train() / transform_test() for the actual
    pipeline.
    """
    df = prepare_features(df)

    low_nan_cols, high_nan_cols = split_columns_by_nan_threshold(df, threshold=0.5)

    df = prune_nonessential_features(df, high_nan_cols)

    # Single outlier-removal pass (duplicate call removed)
    df = filter_outliers(df, outlier_percentages)

    df, _, _ = impute_missing_knn(df)

    return df


def split_data(df: pd.DataFrame, target_col: str = "Churn"):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in input DataFrame.")

    # Only remove rows with missing target values before stratified split.
    before = len(df)
    df = df[df[target_col].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} rows with missing '{target_col}' before split."
        )

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)


def fit_transform_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    raw_df: pd.DataFrame,
    target_variance=0.99,
):
    X_train = X_train.copy()
    X_train["Churn"] = y_train
    # Keep metadata source strictly aligned to current training rows only.
    raw_df = raw_df.loc[X_train.index].copy()

    # --- Step 1: Compute DaysSinceRegistration on train only ---
    X_train, reference_date = compute_days_since_registration(X_train)
    X_train = X_train.drop(columns=["RegistrationDate"], errors="ignore")
    logger.info(f"Training reference_date for DaysSinceRegistration: {reference_date}")

    # --- Step 2: Outlier removal on training data only ---
    X_train = filter_outliers(X_train, outlier_percentages)
    raw_df = raw_df.loc[X_train.index].copy()
    y_train_clean = X_train["Churn"]
    X_train = X_train.drop(columns=["Churn"], errors="ignore")
    X_train["Churn"] = y_train_clean

    # --- Step 3: Target encoding (fit on train labels only) ---
    X_train, country_enc = target_encode(
        X_train, "Country", target_col="Churn", smoothing=30
    )
    X_train, geoip_enc = target_encode(
        X_train, "GeoIP", target_col="Churn", smoothing=20
    )

    # --- Step 3b: One-hot encoding (fit on train only) ---
    X_train = apply_one_hot_encoding(X_train, one_hot_cols)

    # --- Step 4: Drop high-NaN columns, impute low-NaN columns (features only) ---
    y_train_clean = X_train["Churn"].copy()
    X_train_features = X_train.drop(columns=["Churn"], errors="ignore")

    low_nan_cols, high_nan_cols = split_columns_by_nan_threshold(
        X_train_features, threshold=0.5
    )
    X_train_features = prune_nonessential_features(X_train_features, high_nan_cols)
    X_train_features, fitted_imputer, fitted_knn_scaler = impute_missing_knn(
        X_train_features
    )

    # Re-attach target after feature-only preprocessing.
    X_train = X_train_features.copy()
    X_train["Churn"] = y_train_clean.loc[X_train_features.index]

    # --- Step 5: Scale ---
    X_train, fitted_final_scaler = apply_standard_scaler(X_train, target_col="Churn")

    # --- Step 6: Feature selection (train data only) ---
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

    # --- Step 6b: Build columns_types metadata for the UI ---
    one_hot_prefix_map = {prefix: clean_column_name(prefix) for prefix in one_hot_cols}

    one_hot_groups = {}
    for raw_prefix, cleaned_prefix in one_hot_prefix_map.items():
        if raw_prefix not in raw_df.columns:
            logger.warning(
                f"One-hot prefix '{raw_prefix}' not found in raw_df, skipping."
            )
            continue

        # Check at least one encoded column survived feature selection
        surviving_cols = [
            col for col in X_train_clean.columns if col.startswith(raw_prefix + "_")
        ]
        if not surviving_cols:
            logger.info(
                f"All encoded columns for '{raw_prefix}' were dropped during feature selection, skipping."
            )
            continue

        one_hot_groups[cleaned_prefix] = sorted(
            raw_df[raw_prefix].dropna().unique().tolist()
        )

    columns_types = {}

    def map_dtype(dtype) -> str:
        dtype_str = str(dtype)
        if dtype_str.startswith("int"):
            return "int"
        if dtype_str.startswith("float"):
            return "float"
        if dtype_str == "bool":
            return "bool"
        return "str"

    # Scalar columns — skip anything that was one-hot encoded
    for col in X_train_clean.columns:
        if any(col.startswith(raw + "_") for raw in one_hot_prefix_map):
            continue
        source_dtype = raw_df[col].dtype if col in raw_df.columns else X_train_clean[col].dtype
        columns_types[clean_column_name(col)] = {
            "type": map_dtype(source_dtype)
        }

    # One-hot groups — collapse into select or bool
    for prefix, labels in one_hot_groups.items():
        if len(labels) == 1:
            columns_types[f"{prefix} {labels[0]}"] = {"type": "bool"}
            continue
        columns_types[prefix] = {"type": "select", "values": labels}

    logger.info(f"columns_types: {columns_types}")

    # --- Step 7: PCA ---
    pca = PCA(n_components=target_variance, random_state=42)
    X_train_pca_array = pca.fit_transform(X_train_clean)
    n_components = pca.n_components_
    logger.info(
        "PCA n_components: %s, Explained Variance Ratio: %s",
        n_components,
        pca.explained_variance_ratio_.sum(),
    )

    pca_cols = [f"PC{i+1}" for i in range(X_train_pca_array.shape[1])]
    X_train_pca = pd.DataFrame(
        X_train_pca_array, columns=pca_cols, index=X_train_clean.index
    )

    # --- Step 8: SMOTE balancing ---
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_pca, y_train_clean)

    fitted_artifacts = {
        "reference_date": reference_date,
        "country_enc": country_enc,
        "geoip_enc": geoip_enc,
        "one_hot_columns": X_train_clean.columns.tolist(),
        "high_nan_cols": high_nan_cols,
        "low_nan_cols": low_nan_cols,
        "imputer": fitted_imputer,
        "knn_scaler": fitted_knn_scaler,
        "final_scaler": fitted_final_scaler,
        "features_to_drop": features_to_drop,
        "pca": pca,
    }

    return X_train_bal, y_train_bal, fitted_artifacts, columns_types


def transform_test(X_test: pd.DataFrame, fitted_artifacts: dict) -> pd.DataFrame:
    """Applies all fitted transformers to the test set (or new inference data).
    Uses saved artifacts so no test statistics influence any transformation.
    """
    X_test = X_test.copy()

    # Apply fixed reference_date from training — no leakage from test dates
    X_test, _ = compute_days_since_registration(
        X_test, reference_date=fitted_artifacts["reference_date"]
    )
    # Drop raw date column now that the engineered feature has been created
    X_test = X_test.drop(columns=["RegistrationDate"], errors="ignore")

    X_test, _ = target_encode(
        X_test, "Country", encoder=fitted_artifacts["country_enc"]
    )
    X_test, _ = target_encode(X_test, "GeoIP", encoder=fitted_artifacts["geoip_enc"])

    # Apply train-fitted one-hot schema and align columns before downstream transforms.
    X_test = apply_one_hot_encoding(X_test, one_hot_cols)
    expected_after_ohe = fitted_artifacts.get("one_hot_columns")
    if expected_after_ohe:
        for col in expected_after_ohe:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[[c for c in expected_after_ohe if c in X_test.columns]]

    X_test = prune_nonessential_features(X_test, fitted_artifacts["high_nan_cols"])

    # Align to fitted imputer schema (drop extras, add missing) for deterministic transforms.
    imputer = fitted_artifacts["imputer"]
    if hasattr(imputer, "feature_names_in_"):
        imputer_cols = list(imputer.feature_names_in_)
        for col in imputer_cols:
            if col not in X_test.columns:
                X_test[col] = 0

        # Keep non-imputed columns (e.g., bool dummies), but ensure imputer numeric schema matches.
        numeric_cols = X_test.select_dtypes(include=["number"]).columns.tolist()
        extra_numeric = [c for c in numeric_cols if c not in imputer_cols]
        if extra_numeric:
            X_test = X_test.drop(columns=extra_numeric, errors="ignore")

        # Deterministic ordering for downstream transforms.
        remaining_cols = [c for c in X_test.columns if c not in imputer_cols]
        X_test = X_test[imputer_cols + remaining_cols]

    X_test, _, _ = impute_missing_knn(
        X_test,
        imputer=imputer,
        scaler=fitted_artifacts["knn_scaler"],
    )
    X_test, _ = apply_standard_scaler(
        X_test,
        scaler=fitted_artifacts["final_scaler"],
        target_col="Churn",
    )
    X_test = prune_nonessential_features(X_test, fitted_artifacts["features_to_drop"])
    pca = fitted_artifacts["pca"]

    # Ensure columns match exactly the order expected by PCA
    if hasattr(pca, "feature_names_in_"):
        pca_input_cols = list(pca.feature_names_in_)
        for col in pca_input_cols:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[pca_input_cols]

    X_test_pca_array = pca.transform(X_test)

    pca_cols = [f"PC{i+1}" for i in range(X_test_pca_array.shape[1])]
    X_test_pca = pd.DataFrame(X_test_pca_array, columns=pca_cols, index=X_test.index)
    return X_test_pca


def prepare_raw_metadata_source(
    df_raw: pd.DataFrame, target_col: str = TARGET_COL
) -> pd.DataFrame:
    """Builds metadata-only raw dataframe for UI column types/select values.

    This dataframe is intentionally kept close to raw values and is later
    restricted to train rows only.
    """
    raw_df = df_raw.copy()
    raw_df = parse_ip(raw_df)
    raw_df = values_to_nan(raw_df, columns_with_nan_values)
    raw_df = engineer_features(raw_df)
    raw_df = parse_registration_date(raw_df)
    raw_df = raw_df.dropna(subset=[target_col])
    return raw_df


def run_train_test_preprocessing(
    df_raw: pd.DataFrame,
    target_col: str = TARGET_COL,
    target_variance: float = 0.99,
) -> dict:
    """Runs the full train/test preprocessing flow and returns all outputs.

    Returned keys:
    - X_train
    - X_test
    - y_train
    - y_test
    - fitted_artifacts
    - columns_types
    - processed_df
    """
    raw_df_meta = prepare_raw_metadata_source(df_raw, target_col=target_col)

    processed_df = prepare_features(df_raw.copy())
    X_train, X_test, y_train, y_test = split_data(processed_df, target_col=target_col)

    # Train-only metadata source (no category/dtype/value lookups from test rows)
    raw_df_train = raw_df_meta.loc[X_train.index].copy()

    logger.info("Fitting transformations on X_train...")
    X_train_out, y_train_out, fitted_artifacts, columns_types = fit_transform_train(
        X_train,
        y_train,
        raw_df_train,
        target_variance=target_variance,
    )

    logger.info("Applying transformations to X_test using fitted artifacts...")
    X_test_out = transform_test(X_test, fitted_artifacts)

    return {
        "X_train": X_train_out,
        "X_test": X_test_out,
        "y_train": y_train_out,
        "y_test": y_test,
        "fitted_artifacts": fitted_artifacts,
        "columns_types": columns_types,
        "processed_df": processed_df,
    }


def prepare_segmentation_base(
    df_raw: pd.DataFrame,
    reference_date=None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Prepares reusable feature base for segmentation workflows.

    This applies only structural transforms and date-distance engineering,
    intentionally skipping target-aware transformations.
    """
    df_seg = prepare_features(df_raw.copy())
    df_seg, reference_date = compute_days_since_registration(
        df_seg,
        reference_date=reference_date,
    )
    df_seg = df_seg.drop(columns=["RegistrationDate"], errors="ignore")
    return df_seg, reference_date


def main():
    df = load_raw_data()

    outputs = run_train_test_preprocessing(df, target_col=TARGET_COL, target_variance=0.99)
    X_train = outputs["X_train"]
    X_test = outputs["X_test"]
    y_train = outputs["y_train"]
    y_test = outputs["y_test"]
    fitted_artifacts = outputs["fitted_artifacts"]
    columns_types = outputs["columns_types"]
    processed_df = outputs["processed_df"]

    save_processed_data(processed_df, out_dir=PROCESSED_DIR)

    # Save the fitted artifacts for test transformation and future inference
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_artifacts, model_dir / "fitted_artifacts.joblib")
    logger.info(f"Saved fitted artifacts → {model_dir / 'fitted_artifacts.joblib'}")

    save_splits(X_train, X_test, y_train, y_test, out_dir=TRAIN_TEST_DIR)

    # Save the final list of columns after all transformations (except PCA) for reference
    columns_path = model_dir / "final_columns_after_transformations.json"

    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(columns_types, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
