import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    GROUND_PLOT_DRONE_MISSION_MATCHES_FILE,
    GROUND_REFERENCE_PLOTS_FILE,
)

MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 3
TEST_FRACTION = 0.2
OUTPUT_PAIRED_SPLIT_FILE = "/ofo-share/scratch-amritha/tree-species-scratch/processed_02/plot_mission_pairs_with_split.csv"

def hdbscan_spatial_split(plot_gdf, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, test_frac=TEST_FRACTION, seed=35):
    plot_gdf = plot_gdf.to_crs(32610)

    # Extract centroids
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in plot_gdf.geometry])
    centroids = np.array(coords)
    db = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(coords)
    plot_gdf["cluster"] = db.labels_

    # Split into valid and noise
    valid_mask = plot_gdf["cluster"] != -1
    noise_mask = plot_gdf["cluster"] == -1

    valid = plot_gdf[valid_mask].copy()
    noise = plot_gdf[noise_mask].copy()

    # Compute cluster centroids
    cluster_centroids = valid.groupby("cluster").geometry.apply(lambda g: g.unary_union.centroid)
    cluster_centroids_coords = np.array([[pt.x, pt.y] for pt in cluster_centroids])
    cluster_ids = cluster_centroids.index.to_numpy()

    # Build KDTree from cluster centroids
    tree = cKDTree(cluster_centroids_coords)

    # Assign each noise point to nearest cluster
    noise_centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in noise.geometry])
    _, nearest_idx = tree.query(noise_centroids)
    assigned_clusters = cluster_ids[nearest_idx]
    noise["cluster"] = assigned_clusters

    # Merge back
    plot_gdf = pd.concat([valid, noise], ignore_index=True)

    # Compute convex hulls for all clusters
    cluster_hulls = plot_gdf.groupby("cluster").geometry.apply(lambda x: x.unary_union.convex_hull)
    plot_gdf["cluster_convex_hull"] = plot_gdf["cluster"].map(cluster_hulls)

    # Train/test split by cluster
    rng = np.random.default_rng(seed)
    unique_clusters = plot_gdf["cluster"].unique()
    print(f"Number of clusters (including reassigned noise): {len(unique_clusters)}")
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
    ax.set_title("Train-Test split using HDBSCAN")

    # Plot cluster convex hulls as filled polygons colored by split
    hulls = split_gdf[(split_gdf["split"] != "noise")].drop_duplicates(subset=["cluster_convex_hull", "split"])
    for _, row in hulls.iterrows():
        color = "green" if row["split"] == "train" else "red"
        gpd.GeoSeries([row["cluster_convex_hull"]]).plot(ax=ax, facecolor=color, edgecolor=color, alpha=0.2, linewidth=2)

    plt.show()

def main():
    pairs_df = pd.read_csv(GROUND_PLOT_DRONE_MISSION_MATCHES_FILE)
    pairs_df['plot_id'] = pairs_df['plot_id'].apply(lambda x: f"{int(x):04d}")

    plots_gdf = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)[['plot_id', 'geometry']].drop_duplicates(subset='plot_id')
    plots_gdf = gpd.GeoDataFrame(plots_gdf, geometry='geometry')

    # Run HDBSCAN spatial split
    split_plot_gdf = hdbscan_spatial_split(plots_gdf)

    merged_df = pairs_df.merge(split_plot_gdf[['plot_id', 'split']], on='plot_id', how='left')
    print(f"Train plots: {len(merged_df[merged_df['split'] == 'train'])}, Test plots: {len(merged_df[merged_df['split'] == 'test'])}")

    merged_df.to_csv(OUTPUT_PAIRED_SPLIT_FILE, index=False)
    print(f"Saved to: {OUTPUT_PAIRED_SPLIT_FILE}")

    visualize_split(split_plot_gdf)

if __name__ == "__main__":
    main()