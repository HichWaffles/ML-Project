import geoip2.database
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
import ipaddress
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

reader = geoip2.database.Reader("../data/GeoLite2-City.mmdb")

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


def target_encode(df: pd.DataFrame, column: str, smoothing=10) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        column (str): _description_
        smoothing (int, optional): _description_. Defaults to 10.

    Returns:
        pd.DataFrame: _description_
    """
    encoder = TargetEncoder(cols=[column], smoothing=smoothing)
    df[column] = encoder.fit_transform(df[column], df["Churn"])

    return df


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
    return list(to_drop_corr | set(to_drop_vif))


import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def impute_missing_knn(data: pd.DataFrame, target_columns: list = None, n_neighbors=6):
    """
    Standardizes data, performs KNN imputation, and returns data to original scale.

    Parameters:
    - data: pd.DataFrame
    - target_columns: List of columns to impute (defaults to all numeric columns)
    - n_neighbors: Number of neighbors for KNN

    Returns:
    - pd.DataFrame with imputed values
    """
    # 1. Select only numeric data for the imputer
    df_numeric = data.select_dtypes(include=["number"])

    if target_columns is None:
        target_columns = df_numeric.columns.tolist()

    # 2. Scale the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns,
        index=df_numeric.index,
    )

    # 3. Apply KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_scaled = pd.DataFrame(
        imputer.fit_transform(df_scaled),
        columns=df_numeric.columns,
        index=df_numeric.index,
    )

    # 4. Inverse transform
    df_final_numeric = pd.DataFrame(
        scaler.inverse_transform(df_imputed_scaled),
        columns=df_numeric.columns,
        index=df_numeric.index,
    )

    # 5. Merge imputed columns back into a copy of the original dataframe
    df_output = data.copy()
    for col in target_columns:
        df_output[col] = df_final_numeric[col]

    return df_output


def get_irrelevant_features(
    df: pd.DataFrame, target_col: str = "Churn", threshold: float = 0.02
) -> list:
    """
    Identifies features with low correlation to the target variable.

    Parameters:
    - df: pd.DataFrame containing features and target
    - target_col: Name of the target column
    - threshold: Minimum absolute correlation required to keep a feature

    Returns:
    - List of feature names that have low correlation with the target
    """
    correlations = df.corr()[target_col].abs()
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
