import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    GROUND_PLOT_DRONE_MISSION_MATCHES_FILE,
    GROUND_REFERENCE_PLOTS_FILE,
)

EPS_METERS = 200  # Maximum distance between 2 points to be considered in the same cluster
MIN_SAMPLES = 1  # Number of points in a neighborhood for a point to be considered as a core point. 
TEST_FRACTION = 0.2
OUTPUT_PAIRED_SPLIT_FILE = "/ofo-share/scratch-amritha/tree-species-scratch/processed_02/plot_mission_pairs_with_split.csv"

def dbscan_spatial_split(plot_gdf, eps=EPS_METERS, min_samples=MIN_SAMPLES, test_frac=TEST_FRACTION, seed=31):
    """
    Splits GeoDataFrame into spatially separated train and test sets using DBSCAN clustering.
    """
    plot_gdf = plot_gdf.to_crs(32610)

    # Extract centroids of the geometries for clustering
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in plot_gdf.geometry])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

    plot_gdf["cluster"] = db.labels_

    rng = np.random.default_rng(seed)
    unique_clusters = plot_gdf["cluster"].unique()
    print(f"Number of clusters formed: {len(unique_clusters)}")
    test_cluster_count = int(len(unique_clusters) * test_frac)

    test_clusters = rng.choice(unique_clusters, size=test_cluster_count, replace=False)
    plot_gdf["split"] = plot_gdf["cluster"].apply(lambda c: "test" if c in test_clusters else "train")

    return plot_gdf.drop(columns=["cluster"])

def visualize_split(split_gdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    split_gdf[split_gdf["split"] == "train"].plot(ax=ax, color="green", alpha=0.6, edgecolor="k")
    split_gdf[split_gdf["split"] == "test"].plot(ax=ax, color="red", alpha=0.6, edgecolor="k")
    train_patch = mpatches.Patch(color='green', label='Train')
    test_patch = mpatches.Patch(color='red', label='Test')
    plt.legend(handles=[train_patch, test_patch])
    ax.set_title("Train-Test split using DBSCAN")
    plt.show()

def main():
    pairs_df = pd.read_csv(GROUND_PLOT_DRONE_MISSION_MATCHES_FILE)
    pairs_df['plot_id'] = pairs_df['plot_id'].apply(lambda x: f"{int(x):04d}")

    plots_gdf = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)[['plot_id', 'geometry']].drop_duplicates(subset='plot_id')
    plots_gdf = gpd.GeoDataFrame(plots_gdf, geometry='geometry')

    # Run DBSCAN spatial split
    split_plot_gdf = dbscan_spatial_split(plots_gdf)

    merged_df = pairs_df.merge(split_plot_gdf[['plot_id', 'split']], on='plot_id', how='left')
    print(f"Train plots: {len(merged_df[merged_df['split'] == 'train'])}, Test plots: {len(merged_df[merged_df['split'] == 'test'])}")

    # merged_df.to_csv(OUTPUT_PAIRED_SPLIT_FILE, index=False)
    # print(f"Saved to: {OUTPUT_PAIRED_SPLIT_FILE}")

    visualize_split(split_plot_gdf)

if __name__ == "__main__":
    main()