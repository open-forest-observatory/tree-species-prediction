import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import HDBSCAN

import _bootstrap
from configs.path_config import path_config

MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 10
MIN_SAMPLES = 3
TEST_FRACTION = 0.2
SEED = 35


def hdbscan_spatial_clusters(
    plot_gdf,
    min_cluster_size=MIN_CLUSTER_SIZE,
    max_cluster_size=MAX_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
):
    # Convert to meters based CRS
    plot_gdf = plot_gdf.to_crs(32610)
    # Extract centroids of each plot geometry for clustering
    coords = np.column_stack(
        (plot_gdf.geometry.centroid.x, plot_gdf.geometry.centroid.y)
    )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_cluster_size=max_cluster_size,
    ).fit(coords)
    plot_gdf["cluster"] = clusterer.labels_

    # Separate out the valid clustered points and noise (labeled as -1)
    valid = plot_gdf[plot_gdf["cluster"] != -1].copy()
    noise = plot_gdf[plot_gdf["cluster"] == -1].copy()

    # Reassign noise points to the nearest cluster
    # For this, we compute the centroids of each cluster
    # and use a KDTree for efficient nearest neighbor search
    cluster_centroids = valid.groupby("cluster").geometry.apply(
        lambda g: g.unary_union.centroid
    )
    cluster_coords = np.column_stack((cluster_centroids.x, cluster_centroids.y))
    tree = KDTree(cluster_coords)
    cluster_ids = cluster_centroids.index.to_numpy()

    # Compute centroids of noise plot geometries
    noise_coords = np.column_stack(
        (noise.geometry.centroid.x, noise.geometry.centroid.y)
    )
    # Query the nearest cluster centroid for each noise point
    _, nearest_idx = tree.query(noise_coords)
    # Assign the cluster ID of the nearest centroid to the noise points
    noise["cluster"] = cluster_ids[nearest_idx]

    # Merge valid and noise plots back together
    plot_gdf = pd.concat([valid, noise], ignore_index=True)

    return plot_gdf


def assign_train_test_clusters(plot_gdf, test_frac=TEST_FRACTION, seed=SEED):
    # Computer number of plots per cluster
    cluster_sizes = plot_gdf.groupby("cluster").size().reset_index(name="count")
    # frac specifies the fraction of rows to return, 1 means all rows but shuffled
    cluster_sizes = cluster_sizes.sample(frac=1, random_state=seed)

    test_clusters = []
    cumulative_plots = 0
    total_plots = len(plot_gdf)
    # Accumulate clusters for the test set until we approximately reach the desired fraction
    for _, row in cluster_sizes.iterrows():
        if cumulative_plots / total_plots >= test_frac:
            break
        test_clusters.append(row["cluster"])
        cumulative_plots += row["count"]

    # Assign train/test labels
    plot_gdf["split"] = plot_gdf["cluster"].apply(
        lambda c: "test" if c in test_clusters else "train"
    )
    print(
        f"Number of clusters (including reassigned noise): {plot_gdf['cluster'].nunique()}"
    )

    return plot_gdf


def visualize_split(split_gdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    split_gdf[split_gdf["split"] == "train"].plot(
        ax=ax, color="green", alpha=0.6, edgecolor="k"
    )
    split_gdf[split_gdf["split"] == "test"].plot(
        ax=ax, color="red", alpha=0.6, edgecolor="k"
    )
    train_patch = mpatches.Patch(color="green", label="Train")
    test_patch = mpatches.Patch(color="red", label="Test")
    plt.legend(handles=[train_patch, test_patch])
    ax.set_title("Train-Test split using HDBSCAN")

    # Compute convex hulls per cluster
    cluster_hulls = split_gdf.groupby("cluster").geometry.apply(
        lambda x: x.unary_union.convex_hull
    )
    for cluster_id, hull in cluster_hulls.items():
        split = split_gdf[split_gdf["cluster"] == cluster_id]["split"].iloc[0]
        color = "green" if split == "train" else "red"
        gpd.GeoSeries([hull]).plot(
            ax=ax, facecolor=color, edgecolor=color, alpha=0.2, linewidth=2
        )

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Run HDBSCAN and/or train/test split.")

    parser.add_argument(
        "--step",
        choices=["hdbscan", "split", "both"],
        default="both",
        help="Which step to run: hdbscan, split, or both",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to gpkg file with cluster IDs. This is basically the output of hdbscan_spatial_clusters."
        "It's needed if you only want to run the train/test split step.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pairs_df = pd.read_csv(path_config.ground_plot_drone_mission_matches_file)
    pairs_df["plot_id"] = pairs_df["plot_id"].apply(lambda x: f"{int(x):04d}")

    if args.step in ["hdbscan", "both"]:
        plots_gdf = gpd.read_file(path_config.ground_reference_plots_file)[
            ["plot_id", "geometry"]
        ]
        plots_gdf = plots_gdf[plots_gdf["plot_id"].isin(pairs_df["plot_id"])].copy()
        clustered_gdf = hdbscan_spatial_clusters(plots_gdf)
        clustered_gdf.to_file(path_config.hdbscan_clustered_plots)
        print(f"Saved clustered output to {path_config.hdbscan_clustered_plots}")

    if args.step in ["split", "both"]:
        if args.step == "split":
            if args.input is None:
                print(
                    "Error: input file with clustered plots is required for the split step."
                )
                sys.exit(1)
            else:
                clustered_gdf = gpd.read_file(args.input)

        split_gdf = assign_train_test_clusters(clustered_gdf)
        pairs_df = pairs_df.merge(
            split_gdf[["plot_id", "split", "cluster"]], on="plot_id", how="left"
        )
        train_plots = pairs_df[pairs_df["split"] == "train"]["plot_id"].unique()
        test_plots = pairs_df[pairs_df["split"] == "test"]["plot_id"].unique()
        print(f"Train plots: {len(train_plots)}, Test plots: {len(test_plots)}")
        pairs_df.to_csv(path_config.train_test_split_file, index=False)
        print(f"Saved train/test split to {path_config.train_test_split_file}")

        visualize_split(split_gdf)


if __name__ == "__main__":
    main()
