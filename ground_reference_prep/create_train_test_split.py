import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    GROUND_PLOT_DRONE_MISSION_MATCHES_FILE,
    GROUND_REFERENCE_PLOTS_FILE,
    TRAIN_TEST_SPLIT_FILE,
)

MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 3
TEST_FRACTION = 0.2
SEED = 35

def hdbscan_spatial_split(plot_gdf, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, test_frac=TEST_FRACTION, seed=SEED):
    # Convert to meters based CRS
    plot_gdf = plot_gdf.to_crs(32610)

    # Extract centroids of each plot geometry for clustering
    coords = np.column_stack((plot_gdf.geometry.centroid.x, plot_gdf.geometry.centroid.y))
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(coords)
    plot_gdf["cluster"] = clusterer.labels_

    # Separate out the valid clustered points and noise (labeled as -1)
    valid = plot_gdf[plot_gdf["cluster"] != -1].copy()
    noise = plot_gdf[plot_gdf["cluster"] == -1].copy()

    # Reassign noise points to the nearest cluster
    # For this, we compute the centroids of each cluster
    # and use a KDTree for efficient nearest neighbor search
    cluster_centroids = valid.groupby("cluster").geometry.apply(lambda g: g.unary_union.centroid)
    cluster_centroids_coords = np.column_stack((cluster_centroids.x, cluster_centroids.y))
    cluster_ids = cluster_centroids.index.to_numpy()
    tree = KDTree(cluster_centroids_coords)

    # Compute centroids of noise plot geometries
    noise_centroids = np.column_stack((noise.geometry.centroid.x, noise.geometry.centroid.y))
    # Query the nearest cluster centroid for each noise point
    _, nearest_idx = tree.query(noise_centroids)
    # Assign the cluster ID of the nearest centroid to the noise points
    assigned_clusters = cluster_ids[nearest_idx]
    noise["cluster"] = assigned_clusters

    # Merge valid and noise plots back together
    plot_gdf = pd.concat([valid, noise], ignore_index=True)

    # Compute convex hulls for all clusters. This is done only to help visualize the clusters later.
    cluster_hulls = plot_gdf.groupby("cluster").geometry.apply(lambda x: x.unary_union.convex_hull)
    plot_gdf["cluster_convex_hull"] = plot_gdf["cluster"].map(cluster_hulls)

    rng = np.random.default_rng(seed)
    unique_clusters = plot_gdf["cluster"].unique()
    print(f"Number of clusters (including reassigned noise): {len(unique_clusters)}")

    # Calculate number of clusters to assign to test set, using the specified fraction
    test_cluster_count = int(len(unique_clusters) * test_frac)
    # Randomly select clusters for the test set
    test_clusters = rng.choice(unique_clusters, size=test_cluster_count, replace=False)
    plot_gdf["split"] = plot_gdf["cluster"].apply(lambda c: "test" if c in test_clusters else "train")

    return plot_gdf

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

    # Convert plot_id to zero-padded strings
    pairs_df['plot_id'] = pairs_df['plot_id'].apply(lambda x: f"{int(x):04d}")

    # Load ground reference plots, and select plot ID and geometry columns
    plots_gdf = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)[['plot_id', 'geometry']]
    plots_gdf = gpd.GeoDataFrame(plots_gdf, geometry='geometry')

    # Run HDBSCAN spatial split
    split_plot_gdf = hdbscan_spatial_split(plots_gdf)

    # Merge the results with the pairs DataFrame to filter out the 217 plots
    merged_df = pairs_df.merge(split_plot_gdf[['plot_id', 'split', 'cluster']], on='plot_id', how='left')
    print(f"Train plots: {len(merged_df[merged_df['split'] == 'train'])}, Test plots: {len(merged_df[merged_df['split'] == 'test'])}")

    merged_df.to_csv(TRAIN_TEST_SPLIT_FILE, index=False)
    print(f"Saved to: {TRAIN_TEST_SPLIT_FILE}")

    visualize_split(split_plot_gdf)

if __name__ == "__main__":
    main()