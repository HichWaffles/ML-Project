"""
plot_kmeans.py
Plots the actual K-Means customer points in 2D space using PCA.
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import RAW_DATA_PATH, SEGMENTS_DIR, REPORTS_DIR, logger
from src.segment_customers import THEMES, prepare_theme_features

def main():
    if not RAW_DATA_PATH.exists():
        logger.error(f"Cannot find raw data at {RAW_DATA_PATH}")
        return
        
    segments_path = SEGMENTS_DIR / "customer_segments.csv"
    if not segments_path.exists():
        logger.error(f"Cannot find segments at {segments_path}")
        return

    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_segments = pd.read_csv(segments_path)
    
    # Merge segments into raw data
    df = df_raw.merge(df_segments, on="CustomerID", how="inner")
    
    out_dir = REPORTS_DIR / "figures" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for theme, config in THEMES.items():
        logger.info(f"Generating K-Means scatter plot for {theme}...")
        
        # 1. We must run the exact same data preparation to get the accurate feature space
        # (Otherwise, sentinel values like 999 support tickets will completely crush the PCA)
        df_clean = prepare_theme_features(df, theme, config)
        if df_clean.empty:
            continue
            
        segment_col = f"{theme}_Segment"
        if segment_col not in df.columns:
            logger.warning(f"{segment_col} not found in data. Skipping {theme}.")
            continue
            
        # 2. Standardize and apply PCA to reduce to 2 components
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_clean)
        
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)
        
        plot_df = pd.DataFrame(data=pcs, columns=["PC1", "PC2"])
        plot_df["Segment"] = df[segment_col].values
        
        # 3. Plotting
        plt.figure(figsize=(10, 8))
        
        # Define a consistent color palette
        unique_segments = plot_df["Segment"].unique()
        palette = sns.color_palette("Set2", n_colors=len(unique_segments))
        
        sns.scatterplot(
            data=plot_df, 
            x="PC1", y="PC2", 
            hue="Segment", 
            palette=palette,
            alpha=0.6, 
            s=40,
            edgecolor=None
        )
        
        var_explained = pca.explained_variance_ratio_ * 100
        plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1f}% variance)", fontweight='bold')
        plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1f}% variance)", fontweight='bold')
        plt.title(f"K-Means Clusters: {theme} Theme (PCA Projection)", fontsize=16, fontweight='bold', pad=15)
        
        # Adjust legend
        plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        out_file = out_dir / f"{theme.lower()}_kmeans_scatter.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved {out_file.name}")

if __name__ == "__main__":
    main()
