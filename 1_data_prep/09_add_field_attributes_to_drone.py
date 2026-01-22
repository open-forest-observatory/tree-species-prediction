import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from tree_registration_and_matching.eval import match_trees_singlestratum
from tree_registration_and_matching.utils import is_overstory

import _bootstrap
from configs.path_config import path_config

# These datasets have been identified as bad because of low recall
DATASETS_TO_EXCLUDE = [
    "0069_000227_000233",
    "0069_000781_000780",
    "0069_000782_000780",
    "0190_001218_001219",
    "0273_001016_001015",
    "0275_001017_001019",
    "0276_001013_001014",
]


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


def cleanup_field_trees(ground_reference_trees: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # The decay class specifies how severely a dead trees is decaying. At values above decay class 2,
    # it is expected that the stem may be broken. This would cause issues estimating the height from
    # DBH, and likely suggests a tree that will overall not be reconstructed well. Therefore, these
    # trees are dropped prior to matching.
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


def filter_by_live_and_height(tree_gdf, height_column="height"):
    # Drop any dead trees. Note that there may be classes other than "L" (live) in the output
    # but these are assumed to be live as well.
    tree_gdf = tree_gdf.copy()[tree_gdf.live_dead != "D"]
    # Drop any crowns that were less than 10m tall
    tree_gdf = tree_gdf[tree_gdf[height_column] > 10]

    return tree_gdf


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

    stats = []

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
        # Remove short and dead trees
        updated_drone_crowns = filter_by_live_and_height(
            updated_drone_crowns, height_column="height_field"
        )

        report_stats = True

        if report_stats:
            drone_crowns_within_perim = drone_crowns[
                drone_crowns.within(ground_plot_perim.geometry.values[0])
            ]
            drone_trees_live_tall = drone_crowns_within_perim[drone_crowns.height > 10]
            ground_trees_live_tall = filter_by_live_and_height(ground_trees)

            n_matched = len(updated_drone_crowns)
            n_field = len(ground_trees_live_tall)
            n_drone = len(drone_trees_live_tall)
            stats.append((n_matched, n_field, n_drone))

        # We have the number of successfully matched trees here
        # Now we need to figure out the number of *filtered* trees in both sets

        # Only write out crowns if there are more than 10 trees remaining
        if len(updated_drone_crowns) >= 10:
            updated_drone_crowns.to_file(
                Path(
                    path_config.drone_crowns_with_field_attributes,
                    high_quality_dataset + ".gpkg",
                )
            )
    stats = pd.DataFrame(stats, columns=["matched", "field", "drone"])
    stats.to_csv(
        "/ofo-share/repos/david/tree-species-prediction/scratch/matching_stats.csv"
    )
