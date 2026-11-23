import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

import _bootstrap
from configs.path_config import path_config

# Meters-based CRS used for all spatial operations
METRIC_CRS = 32610

# HN polygons that extend beyond their LO polygon by more than this many sq meters trigger a warning
HN_OUTSIDE_LO_TOLERANCE_M2 = 1.0


def load_composite_pairs(composites_gdf):
    """
    Reshape the curated composites file (one row per mission, two missions per composite) into one
    row per high-nadir (HN) / low-oblique (LO) pair.

    Args:
        composites_gdf (gpd.GeoDataFrame): Rows from selected-composites-polygons.gpkg with columns
            composite_id, mission_type ("hn" or "lo"), mission_id, date, geometry

    Returns:
        gpd.GeoDataFrame: One row per composite with mission_id_hn, mission_id_lo, date_hn, date_lo,
            drone_date, hn_lo_date_diff. The geometry is the HN mission polygon.
    """
    # Every composite should be made of exactly one HN and one LO mission
    type_counts = composites_gdf.groupby(["composite_id", "mission_type"]).size()
    if not (type_counts == 1).all() or set(
        composites_gdf["mission_type"].unique()
    ) != {"hn", "lo"}:
        raise ValueError("Each composite must have exactly one 'hn' and one 'lo' mission")

    # Project to a meters-based CRS so areas are in sq meters
    composites_gdf = composites_gdf.to_crs(METRIC_CRS)

    columns = ["composite_id", "mission_id", "date", "geometry"]
    hn = composites_gdf.loc[composites_gdf["mission_type"] == "hn", columns]
    lo = composites_gdf.loc[composites_gdf["mission_type"] == "lo", columns]
    # join on composite_id to get one row per HN/LO pair, with suffixes to distinguish the two missions
    pairs = hn.merge(lo, on="composite_id", suffixes=("_hn", "_lo"))

    # Create GeoSeries for the HN and LO geometries, preserving the CRS
    hn_geom = gpd.GeoSeries(pairs["geometry_hn"], crs=composites_gdf.crs)
    lo_geom = gpd.GeoSeries(pairs["geometry_lo"], crs=composites_gdf.crs)

    # When checking whether a ground plot falls within a pair, we use the HN mission polygon. The HN
    # polygon is expected to be a subset of the LO polygon (the LO mission may extend further out to
    # capture oblique views into the area of interest). In the current composites file, HN pokes
    # past LO in about half of the composites, but only by tiny slivers (at most ~0.23 sq m), so
    # intersecting HN with LO gives identical plot matches. Using HN directly, but warn if this
    # assumption is violated by more than a negligible area in future versions of the file.
    hn_outside_lo_area = hn_geom.difference(lo_geom).area
    violations = pairs.loc[
        hn_outside_lo_area > HN_OUTSIDE_LO_TOLERANCE_M2, "composite_id"
    ].tolist()
    if violations:
        print(
            f"WARNING: HN polygon extends beyond LO polygon by more than "
            f"{HN_OUTSIDE_LO_TOLERANCE_M2} m² for composites: {violations}. "
            "Consider using the HN/LO intersection instead."
        )

    pairs = gpd.GeoDataFrame(
        pairs.drop(columns=["geometry_hn", "geometry_lo"]),
        geometry=hn_geom.values,
        crs=composites_gdf.crs,
    )

    # The drone pair's date is the date of the later of the two missions
    pairs["drone_date"] = pd.to_datetime(pairs[["date_hn", "date_lo"]].max(axis=1))
    # Absolute date difference between the HN and LO missions, used for tie-breaking
    pairs["hn_lo_date_diff"] = (pairs["date_hn"] - pairs["date_lo"]).abs().dt.days

    print("Number of HN/LO composite pairs:", len(pairs))
    return pairs


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

    # Project ground reference plots to the same meters-based CRS as the drone missions
    ground_ref_plots_gdf.to_crs(paired_drone_missions_gdf.crs, inplace=True)

    # Buffer the plots geometry by 40m
    ground_ref_plots_gdf["geometry"] = ground_ref_plots_gdf.buffer(40)

    # Find ground plots that fall within the drone geometry
    joined = gpd.sjoin(
        ground_ref_plots_gdf, paired_drone_missions_gdf, how="inner", predicate="within"
    )

    # Year difference should be <= 8 calendar years apart to get a valid pair
    joined["ground_and_drone_year_diff"] = (
        joined["drone_date"].dt.year - joined["survey_date_parsed"].dt.year
    ).abs()

    # Keep all plots from project NEON2023. Others must satisfy year difference check.
    valid_pairs = joined[
        (joined["project_name"] == "NEON2023")
        | (joined["ground_and_drone_year_diff"] <= 8)
    ].copy()

    # Convert mission IDs to integers for numerical sorting (needed to pick mission with highest ID when breaking ties)
    valid_pairs["mission_id_hn_int"] = valid_pairs["mission_id_hn"].astype(int)

    # Sort so smallest ground_and_drone_year_diff comes first, within that sort by smallest hn_lo_date_diff,
    # then order by highest mission_id_hn (for consistent tie-breaking)
    valid_pairs = valid_pairs.sort_values(
        ["ground_and_drone_year_diff", "hn_lo_date_diff", "mission_id_hn_int"],
        ascending=[True, True, False],  # descending for mission_id_hn_int
    )

    # Ensure each LO mission is matched only once per ground plot
    # For each (mission_id_lo, plot_id), keep only the row with the smallest date difference
    valid_pairs = valid_pairs.drop_duplicates(
        subset=["mission_id_lo", "plot_id"], keep="first"
    )

    # Drop the helper column
    valid_pairs = valid_pairs.drop(columns=["mission_id_hn_int"])

    ground_plot_drone_missions_matches = valid_pairs[
        [
            "plot_id",
            "composite_id",
            "mission_id_hn",
            "mission_id_lo",
            "date_hn",
            "date_lo",
            "survey_date_parsed",
            "ground_and_drone_year_diff",
        ]
    ].reset_index(drop=True)

    return ground_plot_drone_missions_matches


if __name__ == "__main__":
    # Read curated HN/LO drone mission pairs
    composites_gdf = gpd.read_file(path_config.selected_composites_file)
    # Read ground reference plots
    ground_ref_plots_gdf = gpd.read_file(path_config.ground_reference_plots_file)

    paired_drone_missions_gdf = load_composite_pairs(composites_gdf)
    ground_plot_drone_missions_matches = match_ground_plots_with_drone_missions(
        paired_drone_missions_gdf, ground_ref_plots_gdf
    )

    # Save to file
    ground_plot_drone_missions_matches.to_csv(
        path_config.ground_plot_drone_mission_matches_file
    )
