import json
from pathlib import Path

import geopandas as gpd
from tree_registration_and_matching.register_MEE import align_plot
import numpy as np
import shapely
import pandas as pd

import _bootstrap
from configs.path_config import path_config


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


def is_overstory(tree_dataset: gpd.GeoDataFrame):
    """
    Compute which trees are in the overstory based on heights and locations
    https://github.com/open-forest-observatory/ofo-itd-crossmapping/blob/1c35bb20f31013527c35bc56ab7bf5ef5ab1aa72/workflow/30_evaluate-predicted-trees.R#L90

    Args:
        tree_dataset (gpd.GeoDataFrame): The trees represented as points with a height column

    Returns:
        np.array: binary array representing which trees are overstory
    """
    heights = tree_dataset.height.values

    # If no trees are present, return an empty index
    if len(tree_dataset) == 0:
        return np.array([], dtype=bool)

    # Ensure that we have a meters-based projected CRS
    tree_dataset = tree_dataset.to_crs(3310)

    # Compute the difference in heights between different trees. This is the j axis minus the one on
    # the i axis
    height_diffs = heights[np.newaxis, :] - heights[:, np.newaxis]

    # Get a numpy array of coordinates
    tree_points = shapely.get_coordinates(tree_dataset.geometry)

    # Compute the distances between each tree
    dists = cdist(tree_points, tree_points)

    # Compute the threshold distance for tree based on the difference in height
    dist_threshold = height_diffs * 0.1 + 1

    # Is the tree on the i axis within the threshold distance
    is_within_threshold = dists < dist_threshold

    # Is the tree on the i axis shorter than the one on the j axis
    is_shorter = height_diffs > 0

    # Are both conditions met
    shorter_and_under_threshold = np.logical_and(is_within_threshold, is_shorter)

    # Is the tree not occluded by any other trees
    is_overstory = np.logical_not(np.any(shorter_and_under_threshold, axis=1))

    return is_overstory


if __name__ == "__main__":
    # Read the pairings between drone and field plots
    plot_pairings = pd.read_csv(path_config.overlapping_plots_file)
    # List all the detected trees
    detected_tree_folders = sorted(path_config.tree_detections_folder.glob("*"))
    # Read the ground reference trees
    ground_reference_trees = gpd.read_file(path_config.ground_reference_trees_file)
    # Read the plot bounds
    all_plot_bounds = gpd.read_file(path_config.ground_reference_plots_file)

    # Filter out any dead trees
    # TODO consider only keeping "L" ones rather than discarding "D", since there are over 1000 rows
    # with None values
    ground_reference_trees = ground_reference_trees[
        ground_reference_trees.live_dead != "D"
    ]

    # First replace any missing height values with pre-computed allometric values
    nan_height = ground_reference_trees.height.isna()
    ground_reference_trees[nan_height].height = ground_reference_trees[
        nan_height
    ].height_allometric

    # For any remaining missing height values that have DBH, use an allometric equation to compute
    # the height
    nan_height = ground_reference_trees.height.isna()
    # These parameters were fit on paired height, DBH data from this dataset.
    allometric_height_func = lambda x: 1.3 + np.exp(
        -0.3136489123372108 + 0.84623571 * np.log(x)
    )
    # Compute the allometric height and assign it
    allometric_height = allometric_height_func(
        ground_reference_trees[nan_height].dbh.to_numpy()
    )
    ground_reference_trees.loc[nan_height, "height"] = allometric_height

    # Filter out any trees that still don't have height (just 1 in current experiments)
    ground_reference_trees = ground_reference_trees[
        ~ground_reference_trees.height.isna()
    ]
    shifts_per_dataset = {}

    # Iterate over detected trees
    for detected_tree_folder in detected_tree_folders:
        drone_trees = gpd.read_file(Path(detected_tree_folder, "tree_tops.gpkg"))

        # Parse the mission names
        ID = detected_tree_folder.stem
        plot_ID, high_nadir_ID, low_oblique_ID = ID.split("_")

        plot_ID = int(plot_ID.lstrip("0"))
        high_nadir_ID = int(high_nadir_ID.lstrip("0"))
        low_oblique_ID = int(low_oblique_ID.lstrip("0"))

        # Query the correct row
        matching_row = plot_pairings.query(
            "@low_oblique_ID == mission_id_lo and @high_nadir_ID == mission_id_hn and @plot_ID == plot_id"
        )

        if len(matching_row) > 1:
            print(matching_row)
            raise ValueError("Multiple matching rows")
        elif len(matching_row) == 0:
            print(f"Warning: no match for {plot_ID}, {high_nadir_ID}, {low_oblique_ID}")
            continue

        # use the `plot_id` field to determine the corresponding plot ID
        # Then point to that file and run this pipeline
        plot_ID = f"{int(matching_row['plot_id'].values[0]):04}"

        field_trees = ground_reference_trees.query("@plot_ID == plot_id")
        obs_bounds = all_plot_bounds.query("@plot_ID == plot_id")

        # Compute which trees are overstory
        overstory = is_overstory(field_trees)

        # Subset to only overstory trees
        field_trees = field_trees[overstory]

        if len(field_trees) < 5:
            print(f"Fewer than 5 field trees for plot_ID {plot_ID}. Skipping.")
            continue

        # Compute the shift between the field and drone trees
        shifted_field_trees, final_shift, _ = align_plot(
            field_trees=field_trees, drone_trees=drone_trees, obs_bounds=obs_bounds
        )

        # Write out the shifted field trees
        output_file = Path(path_config.shifted_field_trees_folder, ID + ".gpkg")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        shifted_field_trees.to_file(output_file)
        shifts_per_dataset[ID] = list(final_shift)

        shifts_file = Path(
            path_config.shifted_field_trees_folder, "shifts_per_dataset.json"
        )
        with open(shifts_file, "w") as outfile:
            json.dump(shifts_per_dataset, outfile)
