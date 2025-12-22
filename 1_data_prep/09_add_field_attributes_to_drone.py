import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

import _bootstrap
from configs.path_config import path_config

# These datasets have been identified as bad because of either low recall (first seven) or a high
# fraction of hardwood species (remaining three).
DATASETS_TO_EXCLUDE = [
    "0069_000227_000233",
    "0069_000781_000780",
    "0069_000782_000780",
    "0190_001218_001219",
    "0273_001016_001015",
    "0275_001017_001019",
    "0276_001013_001014",
    "0049_000136_000133",
    "0100_000153_000155",
    "0110_000136_000133",
]


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
    possible_pairings = np.logical_and.reduce(
        [above_min_height, below_max_height, below_max_matching_dist]
    )

    # Extract the indices of possible pairings
    possible_pairing_field_inds, possible_paring_drone_inds = np.where(
        possible_pairings
    )
    possible_pairing_inds = np.vstack(
        [possible_pairing_field_inds, possible_paring_drone_inds]
    ).T

    # Extract the distances corresponding to the valid matches
    possible_dists = distance_matrix[
        possible_pairing_field_inds, possible_paring_drone_inds
    ]

    # Sort so the paired indices are sorted, corresponding to the smallest distance pair first
    ordered_by_dist = np.argsort(possible_dists)
    possible_pairing_inds = possible_pairing_inds[ordered_by_dist]

    # Compute the most possible pairs, which is the min of num field and drone trees
    max_valid_matches = np.min(distance_matrix.shape)

    # Record the valid mathces
    matched_field_tree_inds = []
    matched_drone_tree_inds = []

    # Iterate over the indices
    for field_ind, drone_ind in possible_pairing_inds:
        # If niether the field or drone tree has already been matched, this is a valid pairing
        if (field_ind not in matched_field_tree_inds) and (
            drone_ind not in matched_drone_tree_inds
        ):
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
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    drone_crowns: gpd.GeoDataFrame,
    field_perim: gpd.GeoDataFrame,
    field_buffer_dist: float = 10.0,
):
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
    drone_tree_unique_IDs = drone_trees.iloc[
        matched_drone_tree_inds
    ].unique_ID.to_numpy()
    # These two variables, matched_field_trees and drone_tree_unique_IDs, are now ordered in the same way
    # This means corresponding rows should be paired. Effectively, we could add the
    # drone_tree_unique_ID as a column of the field trees and then merge based on that. But we don't
    # want to modify the dataframe, so it's just provided for the `merge` step.

    # Transfer the attributes to the drone trees.
    drone_crowns_with_additional_attributes = pd.merge(
        left=drone_crowns,
        right=matched_field_trees,
        left_on="treetop_unique_ID",
        right_on=drone_tree_unique_IDs,
        how="left",
        suffixes=(
            "_drone",
            "_field",
        ),  # Append these suffixes in cases of name collisions
    )

    return drone_crowns_with_additional_attributes


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


def cleanup_field_trees(ground_reference_trees: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Filter out any dead trees that are decaying
    ground_reference_trees = ground_reference_trees[
        ~(ground_reference_trees.decay_class > 2)
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

    return ground_reference_trees


if __name__ == "__main__":
    # Load the spatial bounds of the field survey, for all plots
    ground_reference_plot_bounds = gpd.read_file(
        path_config.ground_reference_plots_file
    ).to_crs(3310)
    # Load the field reference trees, for all plots
    ground_reference_trees = gpd.read_file(
        path_config.ground_reference_trees_file
    ).to_crs(3310)

    # Load the shifts per dataset dict
    shift_per_dataset = json.load(open(path_config.shift_per_dataset_file, "r"))
    # Load the quality of the field reference trees
    shift_qualities = pd.read_csv(path_config.shift_quality_file)
    # TODO potentially load hardwood fraction

    # Ensure that the height column is filled out and dead trees are removed
    ground_reference_trees = cleanup_field_trees(ground_reference_trees)

    # Extract the dataset names that have high quality shifts
    high_quality_shift_datasets = shift_qualities.loc[
        shift_qualities.Quality.isin([3, 4]), "Dataset"
    ].tolist()
    # Drop the .tif extension from the dataset names
    high_quality_shift_datasets = [x.split(".")[0] for x in high_quality_shift_datasets]

    # Exclude the datasets that are specifically marked for exclusion
    high_quality_shift_datasets = [
        d for d in high_quality_shift_datasets if d not in DATASETS_TO_EXCLUDE
    ]

    # Make the output folder
    Path(path_config.drone_crowns_with_field_attributes).mkdir(exist_ok=True)

    for high_quality_dataset in high_quality_shift_datasets:
        # Extract which field plot this dataset corresponds to
        plot_id = high_quality_dataset.split("_")[0]
        # Identify the perimiter as a single row from the dataframe
        ground_plot_perim = ground_reference_plot_bounds.query("plot_id == @plot_id")
        # Identify the trees for this field plot
        ground_trees = ground_reference_trees.query("plot_id == @plot_id")
        # Extract the shift for this dataset
        shift = shift_per_dataset[high_quality_dataset][0]

        # Apply the shift to the field trees and plot bounds
        ground_trees.geometry = ground_trees.geometry.translate(
            xoff=shift[0], yoff=shift[1]
        )
        ground_plot_perim.geometry = ground_plot_perim.geometry.translate(
            xoff=shift[0], yoff=shift[1]
        )

        # Only include overstory trees
        overstory_mask = is_overstory(ground_trees)
        ground_trees = ground_trees[overstory_mask]

        # Load the detected trees
        drone_trees = gpd.read_file(
            Path(
                path_config.tree_detections_folder,
                high_quality_dataset,
                "tree_tops.gpkg",
            )
        ).to_crs(3310)
        # Load the detected crowns
        drone_crowns = gpd.read_file(
            Path(
                path_config.tree_detections_folder,
                high_quality_dataset,
                "tree_crowns.gpkg",
            )
        ).to_crs(3310)

        updated_drone_crowns = match_field_and_drone_trees(
            field_trees=ground_trees,
            drone_trees=drone_trees,
            drone_crowns=drone_crowns,
            field_perim=ground_plot_perim,
        )

        # Drop any crowns that were not matched
        updated_drone_crowns = updated_drone_crowns.dropna(subset=["species_code"])
        # Drop any dead trees. Note that there may be classes other than "L" (live) in the output
        # but these are assumed to be live as well.
        updated_drone_crowns = updated_drone_crowns[
            updated_drone_crowns.live_dead != "D"
        ]
        # Drop any crowns that were less than 10m tall
        updated_drone_crowns = updated_drone_crowns[
            updated_drone_crowns.height_field > 10
        ]

        # Only write out crowns if there are more than 10 trees remaining
        if len(updated_drone_crowns) >= 10:
            updated_drone_crowns.to_file(
                Path(
                    path_config.drone_crowns_with_field_attributes,
                    high_quality_dataset + ".gpkg",
                )
            )
