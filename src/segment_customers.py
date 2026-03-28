"""
segment_customers.py — Multi-Theme Segmentation
---------------------------------------------------------
Segments customers across multiple independent behavioural themes:

1. Friction (k=4): How much operational overhead they create.
   - Clean, Serial-Canceller, High-Returner, Support-Heavy
2. Explorer (k=2): Breadth of product catalog interaction.
   - Broad-Explorer, Focused-Buyer
3. Timing (k=2): Temporal transaction cadence.
   - Weekend-Shopper, Weekday-Shopper

Each theme is independently clustered (via K-Means on clipped, scaled features)
and all segments are appended to the final output file along with a
unified profile and insights report.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import load_raw_data, logger, RAW_DATA_PATH, SEGMENTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Theme Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Known sentinel values for SupportTicketsCount that mean "missing"
SUPPORT_SENTINELS = {-1, 999}

THEMES = {
    "Friction": {
        "features": [
            "ReturnRatio",
            "CancelledTransactions",
            "NegativeQuantityCount",
            "SupportTicketsCount",
        ],
        "k": 4,
        "naming_rules": [
            ("CancelledTransactions", "Serial-Canceller"),
            ("ReturnRatio", "High-Returner"),
            ("SupportTicketsCount", "High-Support"),
            (None, "Low-Support"),  # Fallback
        ],
    },
    "Explorer": {
        "features": [
            "UniqueProducts",
            "UniqueDescriptions",
            "AvgProductsPerTransaction",
            "UniqueCountries",
        ],
        "k": 2,
        "naming_rules": [
            ("UniqueProducts", "Broad-Explorer"),
            (None, "Focused-Buyer"),
        ],
    },
    "Timing": {
        "features": [
            "PreferredDayOfWeek",
            "PreferredHour",
            "PreferredMonth",
            "WeekendPurchaseRatio",
            "AvgDaysBetweenPurchases",
        ],
        "k": 2,
        "naming_rules": [
            ("WeekendPurchaseRatio", "Weekend-Shopper"),
            (None, "Weekday-Shopper"),
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Core Pipeline Components
# ─────────────────────────────────────────────────────────────────────────────

def prepare_theme_features(df_raw: pd.DataFrame, theme: str, config: dict) -> pd.DataFrame:
    """Extracts, cleans, and bounds features for a specific theme."""
    features = config["features"]
    available = [c for c in features if c in df_raw.columns]
    
    if len(available) < 2:
        logger.warning(f"Theme '{theme}': Need at least 2 features. Found {available}. Skipping.")
        return pd.DataFrame()

    df = df_raw[available].copy()

    # Clean sentinels
    if "SupportTicketsCount" in df.columns:
        df["SupportTicketsCount"] = df["SupportTicketsCount"].replace(
            list(SUPPORT_SENTINELS), np.nan
        )

    # Median Imputation
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # IQR Clipping [1st, 99th pctile]
    for col in df.columns:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        if pd.notna(lo) and pd.notna(hi):
            df[col] = df[col].clip(lo, hi)

    return df


def cluster_theme(df_clean: pd.DataFrame, config: dict, seed: int = 42) -> tuple[pd.Series, dict[int, str]]:
    """Runs KMeans and assigns names based on the centroid ranking rules."""
    k = config["k"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean)

    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(scaled)

    # Calculate centroids on the original, unscaled (but clean/clipped) data for simpler interpretation
    df_with_labels = df_clean.copy()
    df_with_labels["_Cluster"] = labels
    centroids = df_with_labels.groupby("_Cluster").mean()

    names: dict[int, str] = {}
    assigned: set[int] = set()

    for sort_col, segment_name in config["naming_rules"]:
        if sort_col is None:
            # Fallback for remaining clusters
            for cid in range(k):
                if cid not in assigned:
                    names[cid] = segment_name
            break
            
        if sort_col in centroids.columns:
            # Rank clusters descending by this feature
            ranked = centroids[sort_col].sort_values(ascending=False).index.tolist()
            for cid in ranked:
                if cid not in assigned:
                    names[cid] = segment_name
                    assigned.add(cid)
                    break

    return pd.Series(labels, index=df_clean.index), names


def profile_segments(
    df_raw: pd.DataFrame, 
    df_clean: pd.DataFrame,
    labels: pd.Series, 
    names: dict[int, str], 
    theme_name: str,
    feature_cols: list[str]
) -> pd.DataFrame:
    """Builds descriptive stats for segments within a single theme."""
    extra_profile_cols = [
        "Recency", "Frequency", "MonetaryTotal", "AvgBasketValue", 
        "CustomerTenureDays", "SatisfactionScore", "Churn"
    ]
    
    profiles = []
    for cid in sorted(labels.unique()):
        mask = labels == cid
        sub_raw = df_raw.loc[mask]
        sub_clean = df_clean.loc[mask]
        
        row = {
            "Theme": theme_name,
            "Cluster": cid,
            "Segment": names.get(cid, f"Cluster_{cid}"),
            "Size": int(mask.sum()),
            "PctOfTotal": mask.sum() / len(labels) * 100,
        }
        
        # Means for the clustering features (from clean, sentinel-free data)
        for c in feature_cols:
            if c in sub_clean.columns:
                row[f"Avg_{c}"] = sub_clean[c].mean()
                
        # Means for contextual features (from raw data)
        for c in extra_profile_cols:
            if c in sub_raw.columns:
                val = pd.to_numeric(sub_raw[c], errors="coerce").mean()
                row[f"{c}Rate" if c == "Churn" else f"Avg_{c}"] = val
        
        profiles.append(row)
        
    return pd.DataFrame(profiles)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Insights Generator
# ─────────────────────────────────────────────────────────────────────────────

def _friction_insights(row: pd.Series) -> tuple[list[str], list[str]]:
    priority, recs = [], []
    seg = row["Segment"]
    if seg == "Serial-Canceller":
        priority.append("High cancellation load — audit cancellation reasons; likely stock/fulfilment mismatch")
        recs.append("Offer real-time stock confirmation before order completion")
    elif seg == "High-Returner":
        priority.append(f"Return rate {row.get('Avg_ReturnRatio', 0):.0%} — investigate product descriptions/sizing")
        recs.append("Post-return survey to capture root cause")
    elif seg == "High-Support":
        priority.append(f"Avg {row.get('Avg_SupportTicketsCount', 0):.1f} tickets — high support load, but correlates with slighly better retention.")
        recs.append("Build self-service knowledge base for top contact reasons")
    elif seg == "Low-Support":
        recs.append("Silent majority. They do not request help but have standard/higher churn risk (silent churners).")
        recs.append("Use proactive outreach (e.g., NPS surveys) to identify dissatisfaction before they leave.")
    return priority, recs


def _explorer_insights(row: pd.Series) -> tuple[list[str], list[str]]:
    priority, recs = [], []
    seg = row["Segment"]
    if seg == "Broad-Explorer":
        recs.append("Highly varied product views/purchases. Cross-category promotions work best here.")
        recs.append("Feature 'discover new arrivals' content heavily in their communications.")
    elif seg == "Focused-Buyer":
        recs.append("Narrow product scope. Focus on deep-category upsells and high-volume discounts.")
        recs.append("Do not spam with unrelated categories; stick to their known preferences.")
    return priority, recs


def _timing_insights(row: pd.Series) -> tuple[list[str], list[str]]:
    priority, recs = [], []
    seg = row["Segment"]
    if seg == "Weekend-Shopper":
        recs.append("Major activity peak on weekends. Schedule promotional emails for Friday evening or Saturday morning.")
        recs.append("Offer limited-time flash sales that end Sunday night.")
    elif seg == "Weekday-Shopper":
        recs.append("Activity concentrated during the workweek. Target communications during commute hours or lunch breaks.")
        recs.append("B2B/wholesale messaging might resonate better with this group.")
    return priority, recs


def generate_theme_insights(theme_profiles: pd.DataFrame, theme: str) -> dict[int, dict]:
    """Generates recommendation text based on the specific theme logic."""
    insights = {}
    
    for _, row in theme_profiles.iterrows():
        cid = row["Cluster"]
        
        priority, recs = [], []
        if theme == "Friction":
            priority, recs = _friction_insights(row)
        elif theme == "Explorer":
            priority, recs = _explorer_insights(row)
        elif theme == "Timing":
            priority, recs = _timing_insights(row)
            
        insights[cid] = {
            "segment": row["Segment"],
            "size": int(row["Size"]),
            "pct_of_base": float(row["PctOfTotal"]),
            "churn_rate": float(row.get("ChurnRate", 0.0)),
            "priority": priority,
            "recs": recs,
        }
    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {RAW_DATA_PATH}")

    logger.info("=" * 80)
    logger.info("MULTI-THEME CUSTOMER SEGMENTATION PIPELINE")
    logger.info("=" * 80)

    df_raw = load_raw_data(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df_raw):,} customers")

    df_segments = df_raw.copy()
    all_profiles = []
    all_insights = {}

    for theme, config in THEMES.items():
        logger.info(f"\n--- Processing Theme: {theme} ---")
        
        df_clean = prepare_theme_features(df_raw, theme, config)
        if df_clean.empty:
            continue
            
        labels, names = cluster_theme(df_clean, config)
        
        # Append to main frame
        df_segments[f"{theme}_Cluster"] = labels
        df_segments[f"{theme}_Segment"] = labels.map(names)
        
        # Profile
        profiles = profile_segments(df_raw, df_clean, labels, names, theme, list(df_clean.columns))
        all_profiles.append(profiles)
        
        # Churn independence score
        if "Churn" in df_raw.columns:
            r, _ = pointbiserialr(labels.values, df_raw["Churn"].values)
            logger.info(f"  Churn correlation |r| = {abs(r):.4f}")
            
        # Generate internal insights
        all_insights[theme] = generate_theme_insights(profiles, theme)
        
        for cid, nm in sorted(names.items()):
            logger.info(f"  Cluster {cid}: {nm} (n={(labels==cid).sum():,})")

    # Combine profiles
    master_profiles = pd.concat(all_profiles, ignore_index=True)

    # ── Save Outputs ─────────────────────────────────────────────────────────
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Segments CSV
    out_cols = ["CustomerID", "Churn"] if "Churn" in df_segments.columns else ["CustomerID"]
    for theme in THEMES:
        out_cols += [f"{theme}_Cluster", f"{theme}_Segment"]
        
    final_out_cols = [c for c in out_cols if c in df_segments.columns]
    df_segments[final_out_cols].to_csv(SEGMENTS_DIR / "customer_segments.csv", index=False)
    logger.info(f"\n✓ customer_segments.csv ({len(df_segments):,} rows)")

    # 2. Profiles CSV
    master_profiles.to_csv(SEGMENTS_DIR / "cluster_profiles.csv", index=False)
    logger.info(f"✓ cluster_profiles.csv ({len(master_profiles)} rows total)")

    # 3. Insights Report
    report_path = SEGMENTS_DIR / "segment_insights.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-THEME SEGMENTATION INSIGHTS\n")
        f.write("=" * 80 + "\n\n")

        for theme in all_insights:
            f.write(f"\n{'#' * 80}\n")
            f.write(f" THEME: {theme.upper()}\n")
            f.write(f"{'#' * 80}\n")
            
            for cid in sorted(all_insights[theme]):
                ins = all_insights[theme][cid]
                f.write(f"\n  [{ins['segment']}]\n")
                f.write(f"  Size      : {ins['size']:,} ({ins['pct_of_base']:.1f}%)\n")
                if ins["churn_rate"] > 0:
                    f.write(f"  Churn risk: {ins['churn_rate']:.1%}\n")
                    
                if ins["priority"]:
                    f.write("\n    PRIORITY ACTIONS:\n")
                    for a in ins["priority"]:
                        f.write(f"      * {a}\n")
                        
                if ins["recs"]:
                    f.write("\n    RECOMMENDATIONS:\n")
                    for r in ins["recs"]:
                        f.write(f"      - {r}\n")
            f.write("\n")

    logger.info("✓ segment_insights.txt")

    # ── Summary log ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for theme in THEMES:
        logger.info(f"\n[{theme}]")
        theme_profiles = master_profiles[master_profiles["Theme"] == theme]
        for _, row in theme_profiles.sort_values("Size", ascending=False).iterrows():
            churn_str = f"  churn={row.get('ChurnRate', 0)*100:.0f}%"

            if theme == "Friction":
                logger.info(
                    f"  {row['Segment']:<16s}  n={int(row['Size']):5,} ({row['PctOfTotal']:4.1f}%)"
                    f"  returns={row.get('Avg_ReturnRatio', 0):.0%}"
                    f"  cancels={row.get('Avg_CancelledTransactions', 0):.1f}"
                    f"  support={row.get('Avg_SupportTicketsCount', 0):.1f}"
                    f"{churn_str}"
                )
            elif theme == "Explorer":
                logger.info(
                    f"  {row['Segment']:<16s}  n={int(row['Size']):5,} ({row['PctOfTotal']:4.1f}%)"
                    f"  uniq_prods={row.get('Avg_UniqueProducts', 0):.0f}"
                    f"  prods/txn={row.get('Avg_AvgProductsPerTransaction', 0):.1f}"
                    f"{churn_str}"
                )
            elif theme == "Timing":
                logger.info(
                    f"  {row['Segment']:<16s}  n={int(row['Size']):5,} ({row['PctOfTotal']:4.1f}%)"
                    f"  weekend={row.get('Avg_WeekendPurchaseRatio', 0):.0%}"
                    f"  days_btn_txn={row.get('Avg_AvgDaysBetweenPurchases', 0):.1f}"
                    f"{churn_str}"
                )

    logger.info("\n" + "=" * 80)
    logger.info("DONE")


if __name__ == "__main__":
    main()
