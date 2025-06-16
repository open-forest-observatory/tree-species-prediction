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
FIELD_TREE_BOUNDS = (
    "/ofo-share/repos-david/tree-species-prediction/scratch/field_tree_bounds.gpkg"
)


def mean_min_dist(obs_points, pred_points):
    obs_points_np = shapely.get_coordinates(obs_points.geometry)
    pred_points_np = shapely.get_coordinates(pred_points.geometry)
    pred_points_kd_tree = KDTree(pred_points_np)

    _, closest_inds = pred_points_kd_tree.query(obs_points_np)
    closest_pred_points = pred_points_np[closest_inds]

    pairwise_dists = np.linalg.norm(obs_points_np - closest_pred_points, ord=2, axis=1)
    mean_dist = np.mean(pairwise_dists)

    return mean_dist


def find_best_shift(
    obs_points,
    pred_points,
    search_window=50,
    search_increment=2,
    base_shift_x=0,
    base_shift_y=0,
):
    obs_points_np = shapely.get_coordinates(obs_points.geometry)
    pred_points_np = shapely.get_coordinates(pred_points.geometry)
    pred_points_kd_tree = KDTree(pred_points_np)

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

    mean_dists = []

    for shift in tqdm(shifts):
        shifted_obs_points_np = obs_points_np + shift

        _, closest_inds = pred_points_kd_tree.query(shifted_obs_points_np)
        closest_pred_points = pred_points_np[closest_inds]

        pairwise_dists = np.linalg.norm(
            shifted_obs_points_np - closest_pred_points, ord=2, axis=1
        )
        mean_dists.append(np.mean(pairwise_dists))

    if True:
        x = [shift[0, 0] for shift in shifts]
        y = [shift[0, 1] for shift in shifts]

        plt.scatter(x, y, c=mean_dists)
        plt.show()
    best_shift = shifts[np.argmin(mean_dists)]
    return best_shift


def align_plot(
    field_trees_file, drone_trees_file, field_plot_bounds, height_column="score"
):
    field_trees = gpd.read_file(field_trees_file)
    drone_trees = gpd.read_file(drone_trees_file)
    field_bounds = gpd.read_file(field_plot_bounds)
    # TODO consider ensuring that the field trees are in a geospatial CRS
    drone_trees.to_crs(field_trees.crs, inplace=True)
    field_bounds.to_crs(field_trees.crs, inplace=True)

    # Keep only trees above 10m
    drone_trees = drone_trees[drone_trees[height_column] > 10]
    field_trees = field_trees[field_trees[height_column] > 10]

    find_best_shift(obs_points=field_trees, pred_points=drone_trees)


if __name__ == "__main__":
    detected_trees = gpd.read_file(DETECTED_TREES)
    detected_trees.geometry = detected_trees.centroid
    field_trees = detected_trees.copy()
    field_trees.geometry = field_trees.geometry.apply(
        lambda x: translate(x, xoff=-2, yoff=-1)
    )
    field_trees.to_file(FIELD_TREES)

    field_tree_bounds = field_trees.dissolve().convex_hull
    field_tree_bounds.to_file(FIELD_TREE_BOUNDS)

    align_plot(FIELD_TREES, DETECTED_TREES, FIELD_TREE_BOUNDS)

# align_plot = function(plot_id) {
#
#  # Load the two tree maps
#  obs = st_read(file.path(OBSERVED_UNALIGNED_TREES_DIR, paste0(plot_id, ".gpkg")))
#  pred = st_read(file.path(PRELIM_DETECTED_TREES_DIR, paste0(plot_id, ".gpkg")))
#
#  # Load the observed plot bounds
#  obs_bounds = st_read(file.path(OBSERVED_UNALIGNED_PLOTBOUNDS_DIR, paste0(plot_id, ".gpkg")))
#  obs_bounds = obs_bounds |> st_transform(crs = 3310)
#
#  # Prep the two tree maps for alignment (they need a x, y, and z)
#  if (sum(is.na(obs$height)) / nrow(obs) > 0.5) {
#    stop("More than 50% of the observed tree heights are missing for plot_id ", plot_id)
#  }
#  obs = st_transform(obs, crs = 3310) # TODO: make this more general with a latlon-to-utm function
#  obs_coords = st_coordinates(obs)
#  obs = obs |>
#    mutate(x = obs_coords[, 1],
#          y = obs_coords[, 2],
#          z = height)
#  pred = st_transform(pred, st_crs(obs))
#  pred_coords = st_coordinates(pred)
#  pred = pred |>
#    mutate(x = pred_coords[, 1],
#          y = pred_coords[, 2],
#          z = Z)
#
#  obs_sf = obs # Store the original sf object for shifting & export later
#  obs = st_drop_geometry(obs)
#  pred = st_drop_geometry(pred)
#
#  obs = obs |>
#    select(x, y, z)
#  pred = pred |>
#    select(x, y, z)
#
#  # Focus on the larger trees
#  obs = obs |>
#    filter(z > 10)
#  pred = pred |>
#    filter(z > 10)
#
#  # Visualize the two
#  vis2(pred, obs, zoom_to_obs = TRUE)
#
#  shift = find_best_shift(pred,
#                  obs,
#                  obs_bounds = NULL,
#                  objective_fn = obj_mean_dist_to_closest,
#                  parallel = FALSE)
#
#  # Apply this shift to the observed trees and write
#  geom = st_geometry(obs_sf)
#  geom_shifted = geom + c(shift$shift_x, shift$shift_y)
#  st_crs(geom_shifted) = st_crs(geom)
#  obs_shifted = st_set_geometry(obs_sf, geom_shifted)
#  st_write(obs_shifted, file.path(OBSERVED_ALIGNED_TREES_DIR, paste0(plot_id, ".gpkg")), delete_dsn = TRUE)
#
#  # Apply this shift to the plot bounds and write
#  obs_bounds_geom = st_geometry(obs_bounds)
#  obs_bounds_geom_shifted = obs_bounds_geom + c(shift$shift_x, shift$shift_y)
#  st_crs(obs_bounds_geom_shifted) = st_crs(obs_bounds_geom)
#  obs_bounds_shifted = st_set_geometry(obs_bounds, obs_bounds_geom_shifted)
#  st_write(obs_bounds_shifted, file.path(OBSERVED_ALIGNED_PLOTBOUNDS_DIR, paste0(plot_id, ".gpkg")), delete_dsn = TRUE)
#
#  # Visualize
#  obs_shifted_coords = st_coordinates(obs_shifted)
#  obs_shifted = obs_shifted |>
#    mutate(x = obs_shifted_coords[, 1],
#           y = obs_shifted_coords[, 2],
#           z = height)
#  obs_shifted = st_drop_geometry(obs_shifted)
#  obs_shifted = obs_shifted |>
#    select(x, y, z)
#
#  vis2(pred, obs_shifted, zoom_to_obs = TRUE)
#
# }
#
# purrr::walk(plot_ids, align_plot)
#
#
## Poorly aligned are indexes 5, 13, 22, 23
# plot_ids[c(5, 13, 22, 23)]
## 0015, 0046, 0105, 0110
#
## New poorly aligned are 5, 13, 19, 24
# plot_ids[c(5, 13, 19, 24)]
## 0015, 0046, 0100, 0110
