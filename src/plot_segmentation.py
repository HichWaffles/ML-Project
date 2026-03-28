"""
plot_segmentation.py
Generates visual reports from the multi-theme customer segmentation profiles.
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import SEGMENTS_DIR, REPORTS_DIR, logger

def min_max_scale(df, cols):
    """Scales columns 0 to 1 for relative intensity heatmap."""
    res = df.copy()
    for c in cols:
        min_v = df[c].min()
        max_v = df[c].max()
        if max_v > min_v:
            res[c] = (df[c] - min_v) / (max_v - min_v)
        else:
            res[c] = 0.5
    return res

def plot_theme(theme_name: str, df: pd.DataFrame, out_dir: Path):
    # Sort by size to make charts consistent
    df = df.sort_values("Size", ascending=False)
    segments = df["Segment"].tolist()
    
    # ── 1. Size vs Churn Combo Chart ───────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for Size
    bars = ax1.bar(segments, df["Size"], color="#4C72B0", alpha=0.8, label="Segment Size", edgecolor="black", linewidth=1.2)
    ax1.set_xlabel("Segment", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Number of Customers", color="#4C72B0", fontsize=12, fontweight='bold')
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    
    # Line chart for Churn
    if "ChurnRate" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(segments, df["ChurnRate"], color="#C44E52", marker="o", linewidth=3, markersize=10, label="Churn Risk")
        ax2.set_ylabel("Churn Rate", color="#C44E52", fontsize=12, fontweight='bold')
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        ax2.set_ylim(0, max(df["ChurnRate"]) * 1.3 if max(df["ChurnRate"]) > 0 else 1)
        
        # Add percentage labels
        for i, txt in enumerate(df["ChurnRate"]):
            ax2.annotate(f"{txt:.1%}", (i, txt), textcoords="offset points", xytext=(0, 15), ha='center', color='#C44E52', fontweight='bold', fontsize=11)
    
    # Add count labels inside/above bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + (yval*0.02), f"{int(yval):,}", ha="center", va="bottom", color="black", fontsize=10)
        
    plt.title(f"Customer Analysis • {theme_name.upper()} • Volume vs. Churn", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / f"{theme_name.lower()}_size_vs_churn.png", dpi=300)
    plt.close()
    
    # ── 2. Feature Identity Heatmap ─────────────────────────────────────────
    # Dynamically identify core feature columns used in the clustering
    # First, drop any columns that are entirely NaN for this theme
    df = df.dropna(axis=1, how="all")
    
    extra_cols = ["Avg_Recency", "Avg_Frequency", "Avg_MonetaryTotal", "Avg_AvgBasketValue", "Avg_CustomerTenureDays", "Avg_SatisfactionScore"]
    feature_cols = [c for c in df.columns if c.startswith("Avg_") and c not in extra_cols]
    
    if feature_cols:
        heat_df = df.set_index("Segment")[feature_cols]
        # Clean column names for display
        heat_df.columns = [c.replace("Avg_", "") for c in heat_df.columns]
        
        # Normalize between 0 and 1 so the heatmap colormap shows minimums=light, maximums=dark
        normalized_df = min_max_scale(heat_df, heat_df.columns)
        
        # Build the heatmap
        fig, ax = plt.subplots(figsize=(10, max(5, len(segments) * 1.5)))
        sns.heatmap(
            normalized_df, 
            annot=heat_df.round(2), # Text is the raw actual metric
            fmt="g",
            cmap="Blues",
            linewidths=1,
            linecolor="white",
            cbar_kws={'label': 'Relative Intensity within Theme'}
        )
        
        plt.title(f"Customer Analysis • {theme_name.upper()} • Feature DNA", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("")
        plt.xticks(rotation=45, ha="right", fontsize=11)
        plt.yticks(fontsize=12, fontweight='bold', rotation=0)
        
        fig.tight_layout()
        fig.savefig(out_dir / f"{theme_name.lower()}_feature_heatmap.png", dpi=300)
        plt.close()

def main():
    profile_path = SEGMENTS_DIR / "cluster_profiles.csv"
    if not profile_path.exists():
        logger.error(f"Cannot find profiles at {profile_path}")
        return
        
    df = pd.read_csv(profile_path)
    
    out_dir = REPORTS_DIR / "figures" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating plots for {len(df['Theme'].unique())} themes...")
    
    for theme in df["Theme"].unique():
        theme_df = df[df["Theme"] == theme]
        plot_theme(theme, theme_df, out_dir)
        logger.info(f"✓ Saved plots for {theme}")
        
    logger.info(f"All graphs successfully saved to {out_dir}")

if __name__ == "__main__":
    main()
