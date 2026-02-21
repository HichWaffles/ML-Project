import geoip2.database
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
import ipaddress
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from pathlib import Path

db_path = Path(__file__).parent.parent / "data" / "GeoLite2-City.mmdb"

reader = geoip2.database.Reader(str(db_path))

debug = True


def extract_ip_features(ip):
    if pd.isna(ip) or not isinstance(ip, str):
        return pd.Series([np.nan])

    try:
        is_private = ipaddress.ip_address(ip).is_private

        country = is_private and "Private" or reader.city(ip).country.name

        return pd.Series([country])
    except:
        return pd.Series([np.nan])


def target_encode(
    df: pd.DataFrame, column: str, target_col: str = "Churn", smoothing=10, encoder=None
) -> tuple:
    """
    Applies target encoding. Fits a new encoder if none is provided (Train),
    otherwise uses the existing encoder (Test).
    """
    df_encoded = df.copy()

    if encoder is None:
        encoder = TargetEncoder(cols=[column], smoothing=smoothing)
        # Fit and transform on training data
        df_encoded[column] = encoder.fit_transform(
            df_encoded[column], df_encoded[target_col]
        )
    else:
        # Transform only on test data (target_col is ignored here)
        df_encoded[column] = encoder.transform(df_encoded[column])

    return df_encoded, encoder


def get_features_to_destroy(
    df: pd.DataFrame,
    target_cols: list = ["Churn", "ChurnRiskCategory"],
    corr_threshold: float = 0.8,
    use_vif: bool = True,
    vif_threshold: float = 10.0,
) -> list:
    """
    Identifies redundant features to remove based on high pairwise correlation
    and multi-collinearity (VIF).

    Args:
        df (pd.DataFrame): The input dataset containing features and targets.
        target_cols (list, optional): Target columns to exclude from removal.
            The first item is used as the primary target for correlation checks.
        corr_threshold (float, optional): Maximum allowed absolute Pearson correlation
            between two features. Defaults to 0.8.
        use_vif (bool, optional): Whether to perform VIF analysis after correlation
            filtering. Defaults to True.
        vif_threshold (float, optional): Maximum allowed Variance Inflation Factor.
            Defaults to 10.0.

    Returns:
        list: A list of feature column names that should be dropped.
    """

    # 1. Create a features-only dataframe
    # Cast to float immediately to ensure mathematical stability for bools
    features_df = df.select_dtypes(include=["number", "bool"]).astype(float).copy()

    # Drop targets from features
    features_df = features_df.drop(
        columns=[c for c in target_cols if c in features_df.columns], errors="ignore"
    )

    # Fill NaNs temporarily for mathematical stability
    features_df = features_df.fillna(features_df.median())

    # --- PHASE 1: Correlation Analysis ---
    corr_matrix = features_df.corr().abs()
    primary_target = target_cols[0]

    # Safely calculate correlation between features and the primary target only
    # Coerce target to numeric in case it's boolean or object-encoded
    target_series = pd.to_numeric(df[primary_target], errors="coerce")
    churn_corr = features_df.apply(lambda col: col.corr(target_series)).abs()

    to_drop_corr = set()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper.columns:
        high_corr_pairs = upper.index[upper[col] > corr_threshold].tolist()
        for pair in high_corr_pairs:
            # "Destroy" the feature with the weaker relationship to the target
            if churn_corr.get(col, 0) > churn_corr.get(pair, 0):
                to_drop_corr.add(pair)
            else:
                to_drop_corr.add(col)

    # --- PHASE 2: Optional Iterative VIF Analysis ---
    to_drop_vif = []
    if use_vif:
        X_vif = features_df.drop(columns=list(to_drop_corr), errors="ignore")

        while True:
            if X_vif.shape[1] <= 1:
                break

            # Crucial Fix: Add a constant for accurate VIF calculation
            X_vif_const = sm.add_constant(X_vif)

            # Calculate VIF
            vif_values = [
                variance_inflation_factor(X_vif_const.values, i)
                for i in range(X_vif_const.shape[1])
            ]

            vif_series = pd.Series(vif_values, index=X_vif_const.columns)

            # Remove the constant from the series so we don't accidentally drop it
            if "const" in vif_series:
                vif_series = vif_series.drop("const")

            max_vif = vif_series.max()

            # Check if the highest VIF breaches the threshold
            if np.isfinite(max_vif) and max_vif > vif_threshold:
                max_feat = vif_series.idxmax()
                to_drop_vif.append(max_feat)
                X_vif = X_vif.drop(columns=[max_feat])
            else:
                break

    # Return the combined set of all features to be removed
    print(f"Features to drop based on correlation: {to_drop_corr}")
    print(f"Features to drop based on VIF: {to_drop_vif}")
    return list(to_drop_corr | set(to_drop_vif))


def impute_missing_knn(
    data: pd.DataFrame,
    target_columns: list = None,
    n_neighbors=6,
    imputer=None,
    scaler=None,
) -> tuple:
    """
    Performs KNN imputation avoiding data leakage.
    Returns imputed dataframe, fitted imputer, and fitted scaler.
    """
    df_numeric = data.select_dtypes(include=["number"])

    # Exclude prediction target from imputation features to prevent leakage and shape mismatch
    if "Churn" in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=["Churn"])

    if target_columns is None:
        target_columns = df_numeric.columns.tolist()

    # 1. Scale data for KNN (Fit on Train, Transform on Test)
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

    # 2. Impute (Fit on Train, Transform on Test)
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

    # 3. Inverse transform back to original scale
    df_final_numeric = pd.DataFrame(
        scaler.inverse_transform(df_imputed_scaled),
        columns=df_numeric.columns,
        index=df_numeric.index,
    )

    # 4. Merge back into original DataFrame
    df_output = data.copy()
    for col in target_columns:
        if col in df_final_numeric.columns:
            df_output[col] = df_final_numeric[col]

    return df_output, imputer, scaler


def get_irrelevant_features(
    df: pd.DataFrame, target_cols: list = ["Churn"], threshold: float = 0.1
) -> list:
    """
    Identifies features with low correlation to the target variable.

    Parameters:
    - df: pd.DataFrame containing features and target
    - target_cols: List of target column names
    - threshold: Minimum absolute correlation required to keep a feature

    Returns:
    - List of feature names that have low correlation with the target
    """
    correlations = df.corr()[target_cols].abs().mean(axis=1)
    irrelevant_features = correlations[correlations < threshold].index.tolist()
    return irrelevant_features


def split_columns_by_nan_threshold(df: pd.DataFrame, threshold: float = 0.5) -> tuple:
    """
    Splits columns with missing values into two lists based on a percentage threshold.

    Parameters:
    - df: The input DataFrame.
    - threshold: The cutoff point.

    Returns:
    - low_nan_cols: List of columns with NaN share > 0 and <= threshold.
    - high_nan_cols: List of columns with NaN share > threshold.
    """
    # Calculate the fraction of missing values for each column
    nan_shares = df.isna().mean()

    # Filter for columns that actually have missing values
    cols_with_nans = nan_shares[nan_shares > 0]

    # Split based on threshold
    low_nan_cols = cols_with_nans[cols_with_nans <= threshold].index.tolist()
    high_nan_cols = cols_with_nans[cols_with_nans > threshold].index.tolist()

    return low_nan_cols, high_nan_cols


def apply_standard_scaler(
    df: pd.DataFrame, scaler=None, target_col: str = "Churn"
) -> tuple:
    """
    Standardizes continuous numeric columns, explicitly excluding target and ordinals.
    """
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=["number"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Isolate continuous variables (Assuming > 10 unique values means continuous)
    if scaler is None:
        continuous_cols = [
            col for col in numeric_cols if len(df_scaled[col].dropna().unique()) > 10
        ]
        if continuous_cols:
            scaler = StandardScaler()
            df_scaled[continuous_cols] = scaler.fit_transform(
                df_scaled[continuous_cols]
            )
    else:
        # Use columns defined during fit to prevent mismatch on test set
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
    """
    Identifies and replaces outlier values with NaN using Isolation Forest.

    Parameters:
    - df: The feature DataFrame.
    - contamination: The percentage of outliers to remove.
    - target_column: The name of the column to apply Isolation Forest to.
                     If None, uses all numeric columns.
    """
    # 1. Select columns to evaluate
    if target_column is not None:
        df_eval = df[[target_column]]
    else:
        df_eval = df.select_dtypes(include=["number"])

    if df_eval.empty:
        print("Warning: No valid numeric data found for Isolation Forest.")
        return df

    col_names = target_column if target_column is not None else "all numeric columns"
    print(
        f"Running Isolation Forest on {col_names} to replace top {contamination*100}% of outlier rows with NaN..."
    )

    # 2. Fit and Predict
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(df_eval)

    # 3. Replace outlier rows with NaN in the evaluated columns
    outlier_mask = outlier_labels == -1
    df_clean = df.copy()
    df_clean.loc[outlier_mask, df_eval.columns] = np.nan

    print(f"Replaced {outlier_mask.sum()} outlier rows with NaN.")

    return df_clean
