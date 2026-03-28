import ipaddress
import re

import geoip2.database
import numpy as np
import pandas as pd
import statsmodels.api as sm
from category_encoders import TargetEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils.logger import logger
from src.utils.paths import GEOIP_DB_PATH


reader = geoip2.database.Reader(str(GEOIP_DB_PATH))


def extract_ip_features(ip):
    if pd.isna(ip) or not isinstance(ip, str):
        return pd.Series([np.nan])

    try:
        is_private = ipaddress.ip_address(ip).is_private
        country = "Private" if is_private else reader.city(ip).country.name
        return pd.Series([country])
    except Exception:
        return pd.Series([np.nan])


def target_encode(
    df: pd.DataFrame, column: str, target_col: str = "Churn", smoothing=10, encoder=None
) -> tuple:
    """Apply target encoding, fitting encoder when not provided."""
    df_encoded = df.copy()

    if encoder is None:
        encoder = TargetEncoder(cols=[column], smoothing=smoothing)
        df_encoded[column] = encoder.fit_transform(
            df_encoded[column], df_encoded[target_col]
        )
    else:
        df_encoded[column] = encoder.transform(df_encoded[column])

    return df_encoded, encoder


def identify_redundant_features(
    df: pd.DataFrame,
    target_cols: list = None,
    corr_threshold: float = 0.8,
    use_vif: bool = True,
    vif_threshold: float = 10.0,
) -> list:
    if target_cols is None:
        target_cols = ["Churn", "ChurnRiskCategory"]

    features_df = df.select_dtypes(include=["number", "bool"]).astype(float).copy()
    features_df = features_df.drop(
        columns=[c for c in target_cols if c in features_df.columns], errors="ignore"
    )
    features_df = features_df.fillna(features_df.median())

    corr_matrix = features_df.corr().abs()
    primary_target = target_cols[0]
    target_series = pd.to_numeric(df[primary_target], errors="coerce")
    churn_corr = features_df.apply(lambda col: col.corr(target_series)).abs()

    to_drop_corr = set()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper.columns:
        high_corr_pairs = upper.index[upper[col] > corr_threshold].tolist()
        for pair in high_corr_pairs:
            if churn_corr.get(col, 0) > churn_corr.get(pair, 0):
                to_drop_corr.add(pair)
            else:
                to_drop_corr.add(col)

    to_drop_vif = []
    if use_vif:
        X_vif = features_df.drop(columns=list(to_drop_corr), errors="ignore")

        while True:
            if X_vif.shape[1] <= 1:
                break

            X_vif_const = sm.add_constant(X_vif)
            vif_values = [
                variance_inflation_factor(X_vif_const.values, i)
                for i in range(X_vif_const.shape[1])
            ]

            vif_series = pd.Series(vif_values, index=X_vif_const.columns)
            if "const" in vif_series:
                vif_series = vif_series.drop("const")

            max_vif = vif_series.max()
            if np.isfinite(max_vif) and max_vif > vif_threshold:
                max_feat = vif_series.idxmax()
                to_drop_vif.append(max_feat)
                X_vif = X_vif.drop(columns=[max_feat])
            else:
                break

    logger.info(f"Features to drop based on correlation: {list(to_drop_corr)}")
    logger.info(f"Features to drop based on VIF: {to_drop_vif}")
    return list(to_drop_corr | set(to_drop_vif))


def impute_missing_knn(
    data: pd.DataFrame,
    target_columns: list = None,
    n_neighbors=6,
    imputer=None,
    scaler=None,
) -> tuple:
    df_numeric = data.select_dtypes(include=["number"])

    if "Churn" in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=["Churn"])

    if target_columns is None:
        target_columns = df_numeric.columns.tolist()

    if scaler is None:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index,
        )
    else:
        df_scaled = pd.DataFrame(
            scaler.transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index,
        )

    if imputer is None:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed_scaled = pd.DataFrame(
            imputer.fit_transform(df_scaled),
            columns=df_numeric.columns,
            index=df_numeric.index,
        )
    else:
        df_imputed_scaled = pd.DataFrame(
            imputer.transform(df_scaled),
            columns=df_numeric.columns,
            index=df_numeric.index,
        )

    df_final_numeric = pd.DataFrame(
        scaler.inverse_transform(df_imputed_scaled),
        columns=df_numeric.columns,
        index=df_numeric.index,
    )

    df_output = data.copy()
    for col in target_columns:
        if col in df_final_numeric.columns:
            df_output[col] = df_final_numeric[col]

    return df_output, imputer, scaler


def identify_non_contributory_features(
    df: pd.DataFrame, target_cols: list = None, threshold: float = 0.1
) -> list:
    if target_cols is None:
        target_cols = ["Churn"]
    correlations = df.corr()[target_cols].abs().mean(axis=1)
    return correlations[correlations < threshold].index.tolist()


def split_columns_by_nan_threshold(df: pd.DataFrame, threshold: float = 0.5) -> tuple:
    nan_shares = df.isna().mean()
    cols_with_nans = nan_shares[nan_shares > 0]
    low_nan_cols = cols_with_nans[cols_with_nans <= threshold].index.tolist()
    high_nan_cols = cols_with_nans[cols_with_nans > threshold].index.tolist()
    return low_nan_cols, high_nan_cols


def apply_standard_scaler(
    df: pd.DataFrame, scaler=None, target_col: str = "Churn"
) -> tuple:
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=["number"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if scaler is None:
        continuous_cols = [
            col for col in numeric_cols if len(df_scaled[col].dropna().unique()) > 10
        ]
        if continuous_cols:
            scaler = StandardScaler()
            df_scaled[continuous_cols] = scaler.fit_transform(df_scaled[continuous_cols])
    else:
        continuous_cols = list(scaler.feature_names_in_)
        if continuous_cols:
            df_scaled[continuous_cols] = scaler.transform(df_scaled[continuous_cols])

    return df_scaled, scaler


def remove_outliers_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    target_column: str = None,
):
    if target_column is not None:
        df_eval = df[[target_column]]
    else:
        df_eval = df.select_dtypes(include=["number"])

    if df_eval.empty:
        logger.warning("No valid numeric data found for Isolation Forest.")
        return df

    col_names = target_column if target_column is not None else "all numeric columns"
    logger.info(
        f"Running Isolation Forest on {col_names} to replace top {contamination*100}% of outlier rows with NaN..."
    )

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(df_eval)

    outlier_mask = outlier_labels == -1
    df_clean = df.drop(df.index[outlier_mask])

    logger.info(f"Removed {outlier_mask.sum()} outlier rows.")

    return df_clean


def filter_outliers(
    df: pd.DataFrame, outlier_percentages: dict, calc: bool = False
) -> pd.DataFrame:
    if calc:
        for col in df.select_dtypes(include=["number"]).columns:
            if col in outlier_percentages:
                continue
            median = df[col].median()
            mad = (df[col] - median).abs().median()
            if mad > 0:
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                extreme_outliers = (modified_z_scores.abs() > 3.5).sum()
                extreme_pct = extreme_outliers / len(df)
                if extreme_pct > 0:
                    outlier_percentages[col] = extreme_pct

    for col, extreme_pct in outlier_percentages.items():
        logger.info(
            f"Column '{col}' has {extreme_pct:.2%} extreme outliers based on MAD."
        )

    for col, extreme_pct in outlier_percentages.items():
        if col in df.columns:
            df = remove_outliers_isolation_forest(
                df,
                target_column=col,
                contamination=extreme_pct,
            )

    return df


def clean_column_name(col: str) -> str:
    col = col.replace("_", " ")
    col = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", col)
    col = re.sub(r"([A-Z])\s+([A-Z])", r"\1\2", col)
    col = re.sub(r"\s+", " ", col).strip()
    return col


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create common derived features used by preprocessing and segmentation."""
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = np.where(
            df["Frequency"] > 0, df["MonetaryTotal"] / df["Frequency"], 0
        )

    if "Recency" in df.columns and "CustomerTenure" in df.columns:
        df["TenureRatio"] = np.where(
            df["CustomerTenure"] > 0, df["Recency"] / df["CustomerTenure"], 0
        )
    elif "Recency" in df.columns and "CustomerTenureDays" in df.columns:
        df["TenureRatio"] = np.where(
            df["CustomerTenureDays"] > 0,
            df["Recency"] / df["CustomerTenureDays"],
            0,
        )

    if "SupportTicketsCount" in df.columns and "CustomerTenure" in df.columns:
        df["TicketIntensity"] = df["SupportTicketsCount"] / (df["CustomerTenure"] + 1)
    elif "SupportTicketsCount" in df.columns and "CustomerTenureDays" in df.columns:
        df["TicketIntensity"] = df["SupportTicketsCount"] / (
            df["CustomerTenureDays"] + 1
        )

    if "CancelledTrans" in df.columns and "Frequency" in df.columns:
        df["CancellationRate"] = np.where(
            df["Frequency"] > 0, df["CancelledTrans"] / df["Frequency"], 0
        )
    elif "CancelledTerms" in df.columns and "Frequency" in df.columns:
        df["CancellationRate"] = np.where(
            df["Frequency"] > 0, df["CancelledTerms"] / df["Frequency"], 0
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
    return df


def compute_days_since_registration(df: pd.DataFrame, reference_date=None) -> tuple:
    if reference_date is None:
        reference_date = df["RegistrationDate"].max()
    df["DaysSinceRegistration"] = (reference_date - df["RegistrationDate"]).dt.days
    return df, reference_date
