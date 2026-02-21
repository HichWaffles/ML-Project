import pandas as pd
import numpy as np

from src.utils import (
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


def apply_ordinal_encoding(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def drop_unnecessary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=list(set(columns)), errors="ignore")


columns_with_nan_values = {
    "SupportTickets": [-1, 999],
    "SatisfactionScore": [-1, 0, 99],
    "GeoIP": ["Unspecified", "Unknown"],
}


def values_to_nan(df: pd.DataFrame, columns_with_nan_values: dict) -> pd.DataFrame:
    for col, values in columns_with_nan_values.items():
        if col in df.columns:
            df[col] = df[col].replace(values, np.nan)
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


def prepare_features(
    df: pd.DataFrame, target_encoding=True, impute=True
) -> pd.DataFrame:

    df = apply_ordinal_encoding(df, ordinal_mappings)
    df = apply_one_hot_encoding(df, one_hot_cols)

    # I'll get back to the later given that the extracted features mostly already exist.
    df = parse_ip(df)

    df = parse_registration_date(df)

    if target_encoding:
        df = target_encode(df, "Country", smoothing=30)
        df = target_encode(df, "GeoIP", smoothing=20)

    # replace GeoIP with Country if GeoIP is NA
    # df["GeoIP"] = df.apply(
    #     lambda row: row["Country"] if pd.isna(row["GeoIP"]) else row["GeoIP"], axis=1
    # )

    df = values_to_nan(df, columns_with_nan_values)

    low_nan_cols, high_nan_cols = split_columns_by_nan_threshold(df, threshold=0.5)

    # Technically this is useless because Age is extremely none correlated with churn.
    if impute:
        df = impute_missing_knn(df, target_columns=low_nan_cols, n_neighbors=6)

    # These always have to be dropped at the end, otherwise they mess with the correlation and VIF analyses.
    df = drop_unnecessary_columns(df, columns_to_drop + high_nan_cols)

    return df


def preprocess_data(df: pd.DataFrame, target_encoding=True) -> pd.DataFrame:
    df = prepare_features(df, target_encoding=target_encoding)

    redu_cols = get_features_to_destroy(df, use_vif=False)
    irr_cols = get_irrelevant_features(df)

    df = drop_unnecessary_columns(df, redu_cols + irr_cols)

    # May not add NA values past this point.

    return df
