import pandas as pd
import numpy as np

from src.utils import extract_ip_features

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

columns_to_drop = ["CustomerID", "NewsletterSubscribed", "LastLoginIP"]


def apply_ordinal_encoding(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def drop_unnecessary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=columns, errors="ignore")


columns_with_nan_values = {
    "SupportTickets": [-1, 999],
    "SatisfactionScore": [-1, 0, 99],
}


def values_to_nan(df: pd.DataFrame, columns_with_nan_values: dict) -> pd.DataFrame:
    for col, values in columns_with_nan_values.items():
        if col in df.columns:
            df[col] = df[col].replace(values, np.nan)
    return df


def parse_ip(df: pd.DataFrame) -> pd.DataFrame:
    df[["IP_Country", "IP_Timezone", "IP_Continent"]] = df["LastLoginIP"].apply(
        extract_ip_features
    )

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = values_to_nan(df, columns_with_nan_values)

    df = apply_ordinal_encoding(df, ordinal_mappings)
    df = apply_one_hot_encoding(df, one_hot_cols)
    
    # I'll get back to the later given that the extracted features mostly already exist.
    df = parse_ip(df)

    df = drop_unnecessary_columns(df, columns_to_drop)

    # May not add NA values past this point.

    return df
