import geopandas as gpd
import shapely
from scipy.spatial.distance import cdist
import numpy as np


def match_trees_singlestratum(
    field_trees,
    drone_trees,
    search_height_proportion=0.5,
    search_distance_fun_slope=0.1,
    search_distance_fun_intercept=1,
    height_col="Z",
):
    # It seems this is probably this function:
    # https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-detection-accuracy-assessment.R#L164
    pass
    # Compute the pairwise distance matrix (dense, I don't see a way around it)
    # For each (field/drone, need to pick)
    field_tree_points_np = shapely.get_coordinates(field_trees.geometry)
    drone_tree_points_np = shapely.get_coordinates(drone_trees.geometry)

    # consider if this order should be switched
    # This looks correct, it seems like the observed trees are vertical
    distance_matrix = cdist(field_tree_points_np, drone_tree_points_np)

    field_height = np.expand_dims(field_trees[height_col].to_numpy(), axis=1)
    drone_height = np.expand_dims(drone_trees[height_col].to_numpy(), axis=0)

    # Remove trees that are outside of the matching height thresholds
    min_height = field_height * (1 - search_height_proportion)
    max_height = field_height * (1 + search_height_proportion)
    # Remove trees that have too large of a distance from the focal tree
    max_dist = field_height * search_distance_fun_slope + search_distance_fun_intercept

    # Remove impossible matches
    distance_matrix[field_height < min_height] = np.nan
    distance_matrix[field_height > max_height] = np.nan
    distance_matrix[distance_matrix < min_height] = np.nan


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

    perim_buff = field_perim.buffer(10).geometry[0]

    # Consider within vs intersects or other options
    drone_trees = drone_trees[drone_trees.within(perim_buff)]
    drone_crown = drone_crown[drone_crown.intersects(perim_buff)]

    # Maybe filter some of the short trees
    # Compute the full distance matrix or at least the top n matches


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
