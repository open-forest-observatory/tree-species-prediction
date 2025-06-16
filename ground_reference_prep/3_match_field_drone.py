import geopandas as gpd
from shapely.affinity import translate
import numpy as np
import shapely
from scipy.spatial import KDTree
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


DETECTED_TREES = (
    "/ofo-share/repos-david/tree-species-prediction/scratch/detected_trees.gpkg"
)
FIELD_TREES = "/ofo-share/repos-david/tree-species-prediction/scratch/field_trees.gpkg"


def find_best_shift(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    search_window: float = 50,
    search_increment: float = 2,
    base_shift_x: float = 0,
    base_shift_y: float = 0,
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
            The [x, y] shift that should be applied to the observed trees to align them with the
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
        start=base_shift_x - search_window,
        stop=base_shift_x + search_window,
        step=search_increment,
    )
    y_shifts = np.arange(
        start=base_shift_y - search_window,
        stop=base_shift_y + search_window,
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


def align_plot(field_trees_file, drone_trees_file, height_column="score"):
    field_trees = gpd.read_file(field_trees_file)
    drone_trees = gpd.read_file(drone_trees_file)

    # TODO consider ensuring that the field trees are in a geospatial CRS
    drone_trees.to_crs(field_trees.crs, inplace=True)

    # Keep only trees above 10m
    drone_trees = drone_trees[drone_trees[height_column] > 10]
    field_trees = field_trees[field_trees[height_column] > 10]

    best_shift = find_best_shift(
        field_trees=field_trees, drone_trees=drone_trees, search_increment=5
    )

    shifted_field_trees = field_trees.copy()
    shifted_field_trees.geometry = field_trees.geometry.apply(
        lambda x: translate(x, xoff=best_shift[0], yoff=best_shift[1])
    )
    # TODO should this shifted data be written out somewhere?
    f, ax = plt.subplots()
    drone_trees.plot(ax=ax)
    shifted_field_trees.plot(ax=ax)
    plt.show()


if __name__ == "__main__":
    # Create example data by loading real detected trees, then shifting them to simulate field trees
    detected_trees = gpd.read_file(DETECTED_TREES)
    detected_trees.geometry = detected_trees.centroid
    field_trees = detected_trees.copy()
    field_trees.geometry = field_trees.geometry.apply(
        lambda x: translate(x, xoff=-28, yoff=-43)
    )
    field_trees.to_file(FIELD_TREES)

    align_plot(FIELD_TREES, DETECTED_TREES)
