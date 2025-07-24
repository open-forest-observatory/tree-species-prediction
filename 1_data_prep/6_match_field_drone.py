import itertools
import sys
import typing
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import KDTree
from shapely.affinity import translate
from tqdm import tqdm
from spatial_utils.geospatial import ensure_projected_CRS

import _bootstrap
from configs.path_config import path_config


def find_best_shift(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    search_window: float = 50,
    search_increment: float = 2,
    base_shift: typing.Tuple[float] = (0, 0),
    vis: bool = False,
) -> np.array:
    """
    Compute the shift for the observed trees that minimizes the mean distance between observed trees
    and the nearest drone tree.


    Args:
        field_trees (gpd.GeoDataFrame):
            Dataframe of field trees
        drone_trees (gpd.GeoDataFrame):
            Dataframe of drone trees
        search_window (float, optional):
            Distance in meters to perform grid search. Defaults to 50.
        search_increment (float, optional):
            Increment in meters for grid search. Defaults to 2.
        base_shift_x (float, optional):
            Center the grid search around shifting the x of observations this much. Defaults to 0.
        base_shift_y (float, optional):
            Center the grid search around shifting the y of observations this much. Defaults to 0.
        vis (bool, optional):
            Visualize a scatter plot of the mean closest distance to drone trees for each shift.
            Defaults to False.

    Returns:
        np.array:
            The [x, y] shift that should be applied to the field trees to align them with the
            drone trees
    """
    # Extract the numpy (n, 2) coordinates of the points
    # TODO this could include a .centroid call if we want to be more flexible with different input
    # types
    field_tree_points_np = shapely.get_coordinates(field_trees.geometry)
    drone_tree_points_np = shapely.get_coordinates(drone_trees.geometry)

    # Build a KDTree to accelerate nearest neighbor queries
    drone_kd_tree = KDTree(drone_tree_points_np)

    # Build the shifts. Note that our eventual goal is to recover a shift for the observed trees,
    # assuming the drone trees remain fixed
    x_shifts = np.arange(
        start=base_shift[0] - search_window,
        stop=base_shift[0] + search_window,
        step=search_increment,
    )
    y_shifts = np.arange(
        start=base_shift[1] - search_window,
        stop=base_shift[1] + search_window,
        step=search_increment,
    )
    shifts = [
        np.expand_dims(np.array(shift), axis=0)
        for shift in (itertools.product(x_shifts, y_shifts))
    ]

    # Iterate over the shifts and compute the mean distance to the nearest drone tree for each field
    # tree
    mean_dists = []
    for shift in tqdm(shifts):
        # Shift the points
        shifted_field_tree_points_np = field_tree_points_np + shift

        # The KD tree directly returns the distance from the query to the nearest point
        dist_to_closest_point, _ = drone_kd_tree.query(shifted_field_tree_points_np)
        # Record for later
        mean_dists.append(np.mean(dist_to_closest_point))

    if vis:
        # Extract the x and y components of the shifts
        x = [shift[0, 0] for shift in shifts]
        y = [shift[0, 1] for shift in shifts]

        # Create a scatter plot of the shifts versus the quailty of the alignment
        plt.scatter(x, y, c=mean_dists)
        plt.show()

    # Find the shift that produced the lowest mean distance for each field tree
    best_shift = shifts[np.argmin(mean_dists)][0]
    return best_shift


def align_plot(field_trees, drone_trees, height_column="height", vis=False):
    original_field_CRS = field_trees.crs
    # Transform the drone trees to a cartesian CRS if not already
    field_trees = ensure_projected_CRS(field_trees)

    # Ensure that drone trees are in the same CRS
    drone_trees.to_crs(field_trees.crs, inplace=True)


    if not field_trees[height_column].isna().all():
        drone_trees_subset = drone_trees[drone_trees[height_column] > 10]
        field_trees_subset = field_trees[field_trees[height_column] > 10]
    else:
        drone_trees_subset = drone_trees
        field_trees_subset = field_trees

    # First compute a rough shift and then a fine one
    coarse_shift = find_best_shift(
        field_trees=field_trees_subset,
        drone_trees=drone_trees_subset,
        search_increment=1,
        search_window=10,
        vis=True,
    )
    # This is initialized from the coarse shift
    fine_shift = find_best_shift(
        field_trees=field_trees_subset,
        drone_trees=drone_trees_subset,
        search_window=2,
        search_increment=0.2,
        base_shift=coarse_shift,
    )

    print(f"Rough shift: {coarse_shift}, fine shift: {fine_shift}")

    shifted_field_trees = field_trees.copy()
    # Apply the computed shift to the geometry of all field trees
    shifted_field_trees.geometry = shifted_field_trees.geometry.apply(
        lambda x: translate(x, xoff=fine_shift[0], yoff=fine_shift[1])
    )

    # Convert back to the original CRS
    shifted_field_trees.to_crs(original_field_CRS, inplace=True)

    if vis:
        # Plot the aligned data
        f, ax = plt.subplots()
        shifted_field_trees.plot(ax=ax)
        field_trees.plot(ax=ax)
        plt.show()

    return shifted_field_trees


if __name__ == "__main__":
    # Read the pairings between drone and field plots
    plot_pairings = pd.read_csv(path_config.overlapping_plots_file)
    # List all the detected trees
    detected_tree_folders = sorted(path_config.tree_detections_folder.glob("*"))
    # Read the ground reference trees
    ground_reference_trees = gpd.read_file(path_config.ground_reference_trees_file)

    # Filter out any dead trees
    # TODO consider only keeping "L" ones rather than discarding "D", since there are over 1000 rows
    # with None values
    ground_reference_trees = ground_reference_trees[ground_reference_trees.live_dead != "D"]

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

    # Iterate over detected trees
    for detected_tree_folder in detected_tree_folders:
        drone_trees = gpd.read_file(Path(detected_tree_folder, "tree_tops.gpkg"))

        # Parse the mission names
        ID = detected_tree_folder.stem
        _, high_nadir_ID, low_oblique_ID = ID.split("_")

        high_nadir_ID = int(high_nadir_ID.lstrip("0"))
        low_oblique_ID = int(low_oblique_ID.lstrip("0"))

        # Query the correct row
        matching_row = plot_pairings.query(
            "@low_oblique_ID == mission_id_lo and @high_nadir_ID == mission_id_hn"
        )

        if len(matching_row) == 0:
            raise ValueError("Multiple or now matching rows")

        # use the `plot_id` field to determine the corresponding plot ID
        # Then point to that file and run this pipeline
        plot_ID = f"{int(matching_row['plot_id'].values[0]):04}"

        field_trees = ground_reference_trees.query("@plot_ID == plot_id")

        # Compute the shift between the field and drone trees
        shifted_field_trees = align_plot(
            field_trees=field_trees, drone_trees=drone_trees
        )

        # Write out the shifted field trees
        output_file = Path(path_config.shifted_field_trees_folder, ID + ".gpkg")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        shifted_field_trees.to_file(output_file)
