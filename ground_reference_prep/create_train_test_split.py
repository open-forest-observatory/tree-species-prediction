import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (GROUND_PLOT_DRONE_MISSION_MATCHES_FILE,
                       GROUND_REFERENCE_PLOTS_FILE)

N_CLUSTERS = 30  # 30 or 50
TEST_FRACTION = 0.2
RANDOM_SEED = 42
OUTPUT_PAIRED_SPLIT_FILE = "/ofo-share/scratch-amritha/tree-species-scratch/processed_02/plot_mission_pairs_with_split.csv"

def kmeans_spatial_split(plot_gdf, n_clusters=5, test_frac=0.2, seed=42):

    # Project CRS
    plot_gdf = plot_gdf.to_crs(32610)

    # Extract centroids of the geometries for clustering
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in plot_gdf.geometry])

    # Run kmeans
    km = KMeans(n_clusters=n_clusters, random_state=seed).fit(coords)

    # Assign cluster labels to each plot and get unique labels
    plot_gdf['cluster'] = km.labels_
    unique_clusters = np.unique(km.labels_)

    # init random number generator
    rng = np.random.default_rng(seed)

    # Randomly select a subset of clusters for test subset (based on test fraction)
    test_clusters = rng.choice(unique_clusters, size=int(len(unique_clusters) * test_frac), replace=False)

    # Assign 'test' label to plots in test clusters and 'train' to others
    plot_gdf['split'] = plot_gdf['cluster'].apply(lambda c: 'test' if c in test_clusters else 'train')
    return plot_gdf.drop(columns=['cluster'])

def visualize_split(plot_gdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_gdf.plot(ax=ax, column='split', cmap='coolwarm', legend=True, alpha=0.7, edgecolor='k')
    plt.title("Spatial Train-Test Split")
    plt.show()

def main():
    pairs_df = pd.read_csv(GROUND_PLOT_DRONE_MISSION_MATCHES_FILE)
    pairs_df['plot_id'] = pairs_df['plot_id'].apply(lambda x: f"{int(x):04d}")
    plots_gdf = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)[['plot_id', 'geometry']].drop_duplicates(subset='plot_id')
    plots_gdf = gpd.GeoDataFrame(plots_gdf, geometry='geometry')

    split_plot_gdf = kmeans_spatial_split(plots_gdf, n_clusters=N_CLUSTERS, test_frac=TEST_FRACTION, seed=RANDOM_SEED)
    train_count = len(split_plot_gdf[split_plot_gdf['split'] == 'train'])
    test_count = len(split_plot_gdf[split_plot_gdf['split'] == 'test'])
    print(f"Train plots: {train_count}, Test plots: {test_count}")
    merged_df = pairs_df.merge(split_plot_gdf[['plot_id', 'split']], on='plot_id', how='left')

    split_counts = merged_df.groupby('plot_id')['split'].nunique()
    assert split_counts.max() == 1, "Some plots appear in both train and test."

    merged_df.to_csv(OUTPUT_PAIRED_SPLIT_FILE, index=False)

    visualize_split(split_plot_gdf)

if __name__ == "__main__":
    main()