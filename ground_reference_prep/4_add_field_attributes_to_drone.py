import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from spatial_utils.geospatial import ensure_projected_CRS

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    GROUND_REFERENCE_PLOTS_FILE,
    SHIFTED_FIELD_TREES_FOLDER,
    TREE_DETECTIONS_FOLDER,
    DRONE_CROWNS_WITH_FIELD_ATTRIBUTES,
)


# Taken from here:
# https://stackoverflow.com/questions/6430091/efficient-distance-calculation-between-n-points-and-a-reference-in-numpy-scipy
# This is drop-in replacement for scipy.cdist
def cdist(x, y):
    """
    Compute pair-wise distances between points in x and y.

    Parameters:
        x (ndarray): Numpy array of shape (n_samples_x, n_features).
        y (ndarray): Numpy array of shape (n_samples_y, n_features).

    Returns:
        ndarray: Numpy array of shape (n_samples_x, n_samples_y) containing
        the pair-wise distances between points in x and y.
    """
    # Reshape x and y to enable broadcasting
    x_reshaped = x[:, np.newaxis, :]  # Shape: (n_samples_x, 1, n_features)
    y_reshaped = y[np.newaxis, :, :]  # Shape: (1, n_samples_y, n_features)

    # Compute pair-wise distances using Euclidean distance formula
    pairwise_distances = np.sqrt(np.sum((x_reshaped - y_reshaped) ** 2, axis=2))

    return pairwise_distances


def match_trees_singlestratum(
    field_trees,
    drone_trees,
    search_height_proportion=0.5,
    search_distance_fun_slope=0.1,
    search_distance_fun_intercept=1,
    height_col="height",
    vis=False,
):
    # A reimplementation of
    # https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-detection-accuracy-assessment.R#L164
    # Compute the pairwise distance matrix (dense, I don't see a way around it)
    field_tree_points_np = shapely.get_coordinates(field_trees.geometry)
    drone_tree_points_np = shapely.get_coordinates(drone_trees.geometry)

    # consider if this order should be switched
    # This looks correct, it seems like the observed trees are vertical
    distance_matrix = cdist(field_tree_points_np, drone_tree_points_np)

    # Expand so the field trees are a tall matrix and the drone trees are a wide one
    field_height = np.expand_dims(field_trees[height_col].to_numpy(), axis=1)
    drone_height = np.expand_dims(drone_trees[height_col].to_numpy(), axis=0)

    # Compute upper and lower height bounds for matches
    min_drone_height = field_height * (1 - search_height_proportion)
    max_drone_height = field_height * (1 + search_height_proportion)
    # Compute max spatial distances for valid matches
    max_dist = field_height * search_distance_fun_slope + search_distance_fun_intercept

    # Compute which matches fit the criteria using broadcasting to get a matrix representation
    above_min_height = drone_height > min_drone_height
    below_max_height = drone_height < max_drone_height
    below_max_matching_dist = distance_matrix < max_dist

    # Compute which matches fit all three criteria
    possible_pairings = np.logical_and.reduce([above_min_height, below_max_height, below_max_matching_dist])

    # Extract the indices of possible pairings
    possible_pairing_field_inds, possible_paring_drone_inds = np.where(possible_pairings)
    possible_pairing_inds = np.vstack([possible_pairing_field_inds, possible_paring_drone_inds]).T

    # Extract the distances corresponding to the valid matches
    possible_dists = distance_matrix[possible_pairing_field_inds, possible_paring_drone_inds]

    # Sort so the paired indices are sorted, corresponding to the smallest distance pair first
    ordered_by_dist = np.argsort(possible_dists)
    possible_pairing_inds = possible_pairing_inds[ordered_by_dist]

    # Compute the most possible pairs, which is the min of num field and drone trees
    max_valid_matches = np.min(distance_matrix.shape)

    # Record the valid mathces
    matched_field_tree_inds = []
    matched_drone_tree_inds = []

    # Iterate over the indices
    for (field_ind, drone_ind) in possible_pairing_inds:
        # If niether the field or drone tree has already been matched, this is a valid pairing
        if (field_ind not in matched_field_tree_inds) and (drone_ind not in matched_drone_tree_inds):
            # Add the matches to the lists
            matched_field_tree_inds.append(field_ind)
            matched_drone_tree_inds.append(drone_ind)

        # Check to see if all possible trees have been matched. Note, the length of matched field
        # and matched drone inds is the same, so we only need to check one.
        if len(matched_field_tree_inds) == max_valid_matches:
            break

    if vis:
        # Visualize matches
        f, ax = plt.subplots()
        ax.scatter(x=field_tree_points_np[:, 0], y=field_tree_points_np[:, 1], c="r")
        ax.scatter(x=drone_tree_points_np[:, 0], y=drone_tree_points_np[:, 1], c="b")

        ordered_matched_field_trees = field_tree_points_np[matched_field_tree_inds]
        ordered_matched_drone_trees = drone_tree_points_np[matched_drone_tree_inds]
        lines = [
            [tuple(x), tuple(y)]
            for x, y in zip(ordered_matched_field_trees, ordered_matched_drone_trees)
        ]

        from matplotlib import collections as mc

        lc = mc.LineCollection(lines, colors="k", linewidths=2)
        ax.add_collection(lc)

        plt.show()
    return matched_field_tree_inds, matched_drone_tree_inds


def match_field_and_drone_trees(
    field_trees_path: Path,
    drone_trees_path: Path,
    drone_crowns_path: Path,
    field_perim: gpd.GeoDataFrame,
    field_buffer_dist : float = 10.0,
):
    # Load all the data
    field_trees = gpd.read_file(field_trees_path)
    drone_trees = gpd.read_file(drone_trees_path)
    drone_crown = gpd.read_file(drone_crowns_path)

    # Ensure it's all in the same projected CRS
    field_trees = ensure_projected_CRS(field_trees)
    drone_trees = drone_trees.to_crs(field_trees.crs)
    drone_crown = drone_crown.to_crs(field_trees.crs)
    field_perim = field_perim.to_crs(field_trees.crs)

    # Get the buffered perimiter
    perim_buff = field_perim.buffer(field_buffer_dist).geometry.values[0]

    # Consider within vs intersects or other options
    drone_trees = drone_trees[drone_trees.within(perim_buff)]
    drone_trees.index = np.arange(len(drone_trees))

    # Maybe filter some of the short trees
    # Compute the full distance matrix or at least the top n matches
    matched_field_tree_inds, matched_drone_tree_inds = match_trees_singlestratum(
        field_trees=field_trees, drone_trees=drone_trees, vis=False
    )

    # Compute field trees that were matched
    matched_field_trees = field_trees.iloc[matched_field_tree_inds]
    # Drop the geometry from the field trees since we don't want to keep it
    matched_field_trees.drop("geometry", axis=1, inplace=True)
    # Compute the "unique_ID" for matched drone trees. This is a crosswalk with the
    # "treetop_unique_ID" field in the crown polygons
    drone_tree_unique_IDs = drone_trees.iloc[matched_drone_tree_inds].unique_ID.to_numpy()
    # These two variables, matched_field_trees and drone_tree_unique_IDs, are now ordered in the same way
    # This means corresponding rows should be paired. Effectively, we could add the
    # drone_tree_unique_ID as a column of the field trees and then merge based on that. But we don't
    # want to modify the dataframe, so it's just provided for the `merge` step.

    # Transfer the attributes to the drone trees.
    drone_crowns_with_additional_attributes = pd.merge(
        left=drone_crown,
        right=matched_field_trees,
        left_on="treetop_unique_ID",
        right_on=drone_tree_unique_IDs,
        how="left",
        suffixes=("_drone", "_field") # Append these suffixes in cases of name collisions
    )

    return drone_crowns_with_additional_attributes



if __name__ == "__main__":
    # List files
    shifted_field_trees = list(SHIFTED_FIELD_TREES_FOLDER.glob("*"))
    detected_trees = list(TREE_DETECTIONS_FOLDER.glob("*"))

    # Compute the dataset ID without the extension
    field_datasets = set([f.stem for f in shifted_field_trees])
    detected_tree_datasets = set([f.name for f in detected_trees])

    # Find which datasets are present in both sets
    overlapping_datasets = field_datasets.intersection(detected_tree_datasets)

    # Print how many unpaired datasets there are
    field_missing_pairs = len(field_datasets) - len(overlapping_datasets)
    detected_missing_pairs = len(detected_tree_datasets) - len(overlapping_datasets)

    if field_missing_pairs:
        print(
            f"Warning: {field_missing_pairs} field datasets do not have corresponding detected trees"
        )

    if detected_missing_pairs:
        print(
            f"Warning: {detected_missing_pairs} detected datasets do not have corresponding field trees"
        )

    # Load the spatial bounds of the field survey, for all plots
    field_reference_plot_bounds = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)

    # Create the output directory
    DRONE_CROWNS_WITH_FIELD_ATTRIBUTES.mkdir(exist_ok=True, parents=True)
    for dataset in overlapping_datasets:
        # Extract which field plot this dataset corresponds to
        plot_id = dataset.split("_")[0]
        # Identify the perimiter as a single row from the dataframe
        field_perim = field_reference_plot_bounds.query("plot_id == @plot_id")

        # Create the crowns with additional attributes from the field surveyed trees
        updated_drone_crowns = match_field_and_drone_trees(
            field_trees_path=Path(SHIFTED_FIELD_TREES_FOLDER, dataset + ".gpkg"),
            drone_trees_path=Path(TREE_DETECTIONS_FOLDER, dataset, "tree_tops.gpkg"),
            drone_crowns_path=Path(TREE_DETECTIONS_FOLDER, dataset, "tree_crowns.gpkg"),
            field_perim=field_perim,
        )

        # Save the updated crowns, knowing the output directory has already been created
        updated_drone_crowns.to_file(Path(DRONE_CROWNS_WITH_FIELD_ATTRIBUTES, dataset + ".gpkg"))