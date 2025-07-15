import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (DRONE_MISSIONS_WITH_ALT_FILE,
                       GROUND_PLOT_DRONE_MISSION_MATCHES_FILE,
                       GROUND_REFERENCE_PLOTS_FILE)


def classify_mission(row):
    altitude = row["mean_altitude"]
    pitch = row["camera_pitch_derived"]
    terrain_corr = row["flight_terrain_correlation_photogrammetry"]
    front_overlap = row["overlap_front_nominal"]
    side_overlap = row["overlap_side_nominal"]

    # Check for NaNs
    if (
        pd.isna(altitude)
        or pd.isna(pitch)
        or pd.isna(terrain_corr)
        or pd.isna(front_overlap)
        or pd.isna(side_overlap)
    ):
        return "unknown"

    # Must meet terrain fidelity requirement
    if terrain_corr <= 0.75:
        return "low-terrain-fidelity"

    # High-Nadir requirements
    if (
        110 <= altitude <= 150
        and 0 <= pitch <= 10
        and front_overlap >= 90
        and side_overlap >= 80
    ):
        return "high-nadir"

    # Low-Oblique requirements
    if (
        60 <= altitude <= 100
        and 18 <= pitch <= 38
        and front_overlap >= 70
        and side_overlap >= 60
    ):
        return "low-oblique"

    return "unclassified"


# Helper function to get minimum value from comma-separated values
def extract_min_overlap(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str) and "," in val:
        return min(float(x.strip()) for x in val.split(","))
    # Return single values after converting to float
    return float(val)


def pair_drone_missions(drone_missions_gdf):
    """
    Classify drone missions into high-nadir and low-oblique based on criteria,
    then pair them based on spatial overlap and date of collection.

    Args:
        drone_missions_gdf (gpd.GeoDataFrame): GeoDataFrame containing drone missions metadata

    Returns:
        gpd.GeoDataFrame: Paired drone missions with spatial and temporal criteria met
    """
    # Convert columns to numeric, handling errors and occurances of double overlap values
    drone_missions_gdf["overlap_front_nominal"] = drone_missions_gdf[
        "overlap_front_nominal"
    ].apply(extract_min_overlap)
    drone_missions_gdf["overlap_side_nominal"] = drone_missions_gdf[
        "overlap_side_nominal"
    ].apply(extract_min_overlap)
    drone_missions_gdf["camera_pitch_derived"] = pd.to_numeric(
        drone_missions_gdf["camera_pitch_derived"], errors="coerce"
    )
    drone_missions_gdf["flight_terrain_correlation_photogrammetry"] = pd.to_numeric(
        drone_missions_gdf["flight_terrain_correlation_photogrammetry"], errors="coerce"
    )

    # Classify high-nadir and low-oblique missions
    drone_missions_gdf["mission_type"] = drone_missions_gdf.apply(
        classify_mission, axis=1
    )

    print(
        "Number of missions classified as High-Nadir: ",
        len(drone_missions_gdf[drone_missions_gdf["mission_type"] == "high-nadir"]),
    )
    print(
        "Number of missions classified as Low-Oblique: ",
        len(drone_missions_gdf[drone_missions_gdf["mission_type"] == "low-oblique"]),
    )

    # Next, we need to create pairs of high-nadir and low-oblique missions based on year
    # Convert date column to datetime
    drone_missions_gdf["earliest_date_derived"] = pd.to_datetime(
        drone_missions_gdf["earliest_date_derived"], errors="coerce"
    )

    # Drop rows with unknown mission_type or missing date
    valid_missions = drone_missions_gdf[
        drone_missions_gdf["mission_type"].isin(["high-nadir", "low-oblique"])
        & drone_missions_gdf["earliest_date_derived"].notna()
    ].copy()

    # Split into two groups
    high_nadir = valid_missions[valid_missions["mission_type"] == "high-nadir"].copy()
    low_oblique = valid_missions[valid_missions["mission_type"] == "low-oblique"].copy()

    # Create all possible pairs that spatially overlap
    # Note: after this step, keys from high-nadir will be suffixed with _1 and low-oblique with _2
    paired = gpd.overlay(
        high_nadir, low_oblique, how="intersection", keep_geom_type=False
    )

    # Filter pairs by date of collection.
    # Condition: needs to be collected within same calendar year
    def is_valid_pair(row):
        same_year = (
            row["earliest_date_derived_1"].year == row["earliest_date_derived_2"].year
        )
        return same_year

    paired_valid = paired[paired.apply(is_valid_pair, axis=1)].reset_index(drop=True)

    # Force include pair 337 & 338
    forced_pair = paired[(paired["mission_id_1"] == "000337") & (paired["mission_id_2"] == "000338")]
    paired_valid = pd.concat([paired_valid, forced_pair], ignore_index=True)

    # Calculate absolute date difference
    paired_valid["date_diff_days"] = (
        (
            paired_valid["earliest_date_derived_1"]
            - paired_valid["earliest_date_derived_2"]
        )
        .abs()
        .dt.days
    )

    # Sort so pairs with smaller date differences b/w missions come first
    paired_valid_sorted = paired_valid.sort_values("date_diff_days")

    # Only use each low-oblique mission once, retaining the pair with the closest (in time) nadir mission to it
    # Note: This can have the same high-nadir mission matched to multiple low-oblique missions
    paired_drone_missions_gdf = paired_valid_sorted.drop_duplicates(
        subset="mission_id_2", keep="first"
    )

    return paired_drone_missions_gdf


def match_ground_plots_with_drone_missions(
    paired_drone_missions_gdf, ground_ref_plots_gdf
):
    """
    Match ground reference plots with drone missions pairs based on spatial overlap and date.

    Args:
        paired_drone_missions_gdf (gpd.GeoDataFrame): GeoDataFrame containing paired drone missions
        ground_ref_plots_gdf (gpd.GeoDataFrame): GeoDataFrame containing ground reference plots

    Returns:
        gpd.GeoDataFrame: Ground reference plots matched with drone missions
    """

    # Convert survey_date to datetime, handling various formats
    def parse_survey_date(val):
        s = str(int(val)) if pd.notna(val) else ""
        if len(s) == 4:
            return pd.to_datetime(s + "-07-01")
        elif len(s) == 6:
            return pd.to_datetime(s, format="%Y%m")
        elif len(s) == 8:
            return pd.to_datetime(s, format="%Y%m%d")
        return pd.NaT

    ground_ref_plots_gdf["survey_date_parsed"] = ground_ref_plots_gdf[
        "survey_date"
    ].apply(parse_survey_date)

    # Convert to GeoDataFrame with geometry column as intersection of drone mission geometries
    paired_drone_missions_gdf.to_crs(32610, inplace=True)  # Project to meters-based CRS

    # Set drone_date to the later date of the two missions in the pair
    paired_drone_missions_gdf["drone_date"] = pd.to_datetime(
        paired_drone_missions_gdf[
            ["earliest_date_derived_1", "earliest_date_derived_2"]
        ].max(axis=1)
    )

    # Project ground reference plots to the same CRS as the drone missions
    ground_ref_plots_gdf.to_crs(paired_drone_missions_gdf.crs, inplace=True)

    # Buffer the plots geometry by 40m
    ground_ref_plots_gdf["geometry"] = ground_ref_plots_gdf.buffer(40)

    # Find ground plots that fall within the drone geometry
    joined = gpd.sjoin(
        ground_ref_plots_gdf, paired_drone_missions_gdf, how="inner", predicate="within"
    )

    # Year difference should be <= 8 calendar years apart to get a valid pair
    joined["year_diff"] = (
        joined["drone_date"].dt.year - joined["survey_date_parsed"].dt.year
    ).abs()
    
    # Keep all plots from project NEON2023. Others must satisfy year difference check.
    valid_pairs = joined[(joined["project_name"] == "NEON2023") | (joined["year_diff"] <= 8)]

    # Rename _1 to _hn and _2 to _lo
    paired_columns_rename = {
        "mission_id_1": "mission_id_hn",
        "mission_id_2": "mission_id_lo",
        "earliest_date_derived_1": "earliest_date_derived_hn",
        "earliest_date_derived_2": "earliest_date_derived_lo",
    }
    valid_pairs.rename(columns=paired_columns_rename, inplace=True)

    ground_plot_drone_missions_matches = valid_pairs[
        [
            "plot_id",
            "mission_id_hn",
            "mission_id_lo",
            "earliest_date_derived_hn",
            "earliest_date_derived_lo",
            "survey_date_parsed",
            "year_diff",
        ]
    ].reset_index(drop=True)

    return ground_plot_drone_missions_matches


if __name__ == "__main__":
    # Read file with all drone missions metadata, including computed altitude values
    drone_missions_gdf = gpd.read_file(DRONE_MISSIONS_WITH_ALT_FILE)
    # Read ground reference plots
    ground_ref_plots_gdf = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)

    paired_drone_missions_gdf = pair_drone_missions(drone_missions_gdf)
    ground_plot_drone_missions_matches = match_ground_plots_with_drone_missions(
        paired_drone_missions_gdf, ground_ref_plots_gdf
    )

    # Save to file
    ground_plot_drone_missions_matches.to_csv(GROUND_PLOT_DRONE_MISSION_MATCHES_FILE)
