import geopandas as gpd
import shapely
import numpy as np
from pathlib import Path
from spatial_utils.geospatial import ensure_projected_CRS
import matplotlib.pyplot as plt


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
):
    # It seems this is probably this function:
    # https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-detection-accuracy-assessment.R#L164
    # Compute the pairwise distance matrix (dense, I don't see a way around it)
    # For each (field/drone, need to pick)
    field_tree_points_np = shapely.get_coordinates(field_trees.geometry)
    drone_tree_points_np = shapely.get_coordinates(drone_trees.geometry)

    # consider if this order should be switched
    # This looks correct, it seems like the observed trees are vertical
    distance_matrix = cdist(field_tree_points_np, drone_tree_points_np)

    # Expand so the field trees are a tall matrix and the drone trees are a wide one
    field_height = np.expand_dims(field_trees[height_col].to_numpy(), axis=1)
    drone_height = np.expand_dims(drone_trees[height_col].to_numpy(), axis=0)

    # Remove trees that are outside of the matching height thresholds
    min_drone_height = field_height * (1 - search_height_proportion)
    max_drone_height = field_height * (1 + search_height_proportion)
    # Remove trees that have too large of a distance from the focal tree
    max_dist = field_height * search_distance_fun_slope + search_distance_fun_intercept

    # Remove impossible matches
    distance_matrix[drone_height < min_drone_height] = np.nan
    distance_matrix[drone_height > max_drone_height] = np.nan
    distance_matrix[distance_matrix > max_dist] = np.nan

    # Get the inds of the field trees, starting with the tallest ones

    matched_field_tree_inds = []
    matched_drone_tree_inds = []

    for field_ind in np.argsort(-np.squeeze(field_height)):
        row = distance_matrix[field_ind]
        # If all nan slice, then no matches are available
        if np.all(np.isnan(row)):
            continue

        # Compute the lowest non-nan value, indicating the closest valid tree
        drone_ind = np.nanargmin(row)

        # Add these matches to the lists
        matched_field_tree_inds.append(field_ind)
        matched_drone_tree_inds.append(drone_ind)

        # Ensure this matched drone tree does not match with any other field trees
        distance_matrix[:, drone_ind] = np.nan

    matched_field_tree_inds = np.array(matched_field_tree_inds)
    matched_drone_tree_inds = np.array(matched_drone_tree_inds)

    if True:
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
    field_trees_path,
    drone_trees_path,
    drone_crowns_path,
    field_perim_file,
):
    field_trees = gpd.read_file(field_trees_path)
    drone_trees = gpd.read_file(drone_trees_path)
    drone_crown = gpd.read_file(drone_crowns_path)
    field_perim = gpd.read_file(field_perim_file)

    field_trees = ensure_projected_CRS(field_trees)
    drone_trees = drone_trees.to_crs(field_trees.crs)
    drone_crown = drone_crown.to_crs(field_trees.crs)
    field_perim = field_perim.to_crs(field_trees.crs)

    perim_buff = field_perim.buffer(10).geometry[0]

    # Consider within vs intersects or other options
    drone_trees = drone_trees[drone_trees.within(perim_buff)]
    drone_crown = drone_crown[drone_crown.intersects(perim_buff)]

    # Maybe filter some of the short trees
    # Compute the full distance matrix or at least the top n matches
    match_trees_singlestratum(field_trees=field_trees, drone_trees=drone_trees)


if __name__ == "__main__":
    DETECTED_TREES_FOLDER = Path(
        "/ofo-share/repos-david/tree-species-prediction/scratch/detected_trees"
    )
    match_field_and_drone_trees(
        field_trees_path=Path(DETECTED_TREES_FOLDER, "field_trees.gpkg"),
        drone_trees_path=Path(DETECTED_TREES_FOLDER, "tree_tops.gpkg"),
        drone_crowns_path=Path(DETECTED_TREES_FOLDER, "tree_crowns.gpkg"),
        field_perim_file=Path(DETECTED_TREES_FOLDER, "field_bounds.gpkg"),
    )

#
#  # Run matching and filter to only matched trees
#  matches = match_trees_singlestratum(trees_field_foc,
#                                      trees_drone_foc,
#                                      search_height_proportion = 0.5,
#                                      search_distance_fun_slope = 0.1,
#                                      search_distance_fun_intercept = 1)
#
#  matches = matches |>
#    filter(!is.na(final_predicted_tree_match_id))
#
#  ## Take the crown polygons and look up the species of the matched field tree
#  # First get only the columns we need from the field tree data
#  trees_field_foc_simp = matches |>
#    st_drop_geometry() |>
#    select(observed_tree_id,
#           species_observed = species,
#           height_observed = ht_top,
#           percent_green_observed = pct_current_green,
#           stem_map_name,
#           predicted_tree_id = final_predicted_tree_match_id) |>
#    mutate(live_observed = as.numeric(percent_green_observed) > 0,
#           percent_green_observed = as.numeric(percent_green_observed),
#           fire = fire_name_foc)
#
#  #  Join the field tree data to the drone crown polygons, also pull in the photogrammetry tree height (from the treetop points)
#  crowns_drone_foc_w_field_data = crowns_drone_foc |>
#    inner_join(trees_field_foc_simp, by = "predicted_tree_id") |>
#    left_join(trees_drone_foc |> st_drop_geometry(), by = join_by(predicted_tree_id, stem_map_name)) |>
#    rename(height_chm = height)
#
#  # Bind onto running data frame
#  crowns_drone_w_field_data = rbind(crowns_drone_w_field_data, crowns_drone_foc_w_field_data)
