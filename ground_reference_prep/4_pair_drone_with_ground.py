import geopandas as gpd
import pandas as pd
import numpy as np


DRONE_MISSIONS_WITH_ALT = "/ofo-share/scratch-amritha/tree-species-scratch/ofo-all-missions-metadata-with-altitude.gpkg"
GROUND_REFERENCE_PLOTS = "/ofo-share/species-prediction-project/raw/ground-reference/ofo_ground-reference_plots.gpkg"
OUTPUT_SAVE_PATH = "/ofo-share/scratch-amritha/tree-species-scratch/ground-reference-drone-missions-pairs.gpkg"

def classify_mission(row):
    altitude = row['mean_altitude']
    pitch = row['camera_pitch_derived']
    terrain_corr = row['flight_terrain_correlation_photogrammetry']
    front_overlap = row['overlap_front_nominal']
    side_overlap = row['overlap_side_nominal']

    # Check for NaNs
    if pd.isna(altitude) or pd.isna(pitch) or pd.isna(terrain_corr) \
       or pd.isna(front_overlap) or pd.isna(side_overlap):
        return 'unknown'

    # Must meet terrain fidelity requirement
    if terrain_corr <= 0.75:
        return 'low-terrain-fidelity'

    # High-Nadir requirements
    if (110 <= altitude <= 150 and 0 <= pitch <= 10 and
        front_overlap >= 90 and side_overlap >= 80):
        return 'high-nadir'

    # Low-Oblique requirements
    if (60 <= altitude <= 100 and 18 <= pitch <= 38 and
        front_overlap >= 70 and side_overlap >= 60):
        return 'low-oblique'

    return 'unclassified'


# Helper function to get minimum value from comma-separated values
def extract_min_overlap(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str) and ',' in val:
        return min(float(x.strip()) for x in val.split(','))
    try:
        return float(val)
    except ValueError:
        return np.nan
    

def pair_drone_missions_and_ground_plots(drone_missions_gdf, ground_ref_plots_gdf):
    """ 
    First create high-nadir and low-oblique drone mission pairs based on criteria, 
    then pair them with ground reference plots.

    Args:
        drone_missions_gdf (gpd.GeoDataFrame): GeoDataFrame containing drone missions metadata
        ground_ref_plots_gdf (gpd.GeoDataFrame): GeoDataFrame containing ground reference plots
    """
    # Convert columns to numeric, handling errors and occurances of double overlap values
    drone_missions_gdf['overlap_front_nominal'] = drone_missions_gdf['overlap_front_nominal'].apply(extract_min_overlap)
    drone_missions_gdf['overlap_side_nominal'] = drone_missions_gdf['overlap_side_nominal'].apply(extract_min_overlap)
    drone_missions_gdf['camera_pitch_derived'] = pd.to_numeric(drone_missions_gdf['camera_pitch_derived'], errors='coerce')
    drone_missions_gdf['flight_terrain_correlation_photogrammetry'] = pd.to_numeric(
        drone_missions_gdf['flight_terrain_correlation_photogrammetry'], errors='coerce'
    )

    # Classify high-nadir and low-oblique missions
    drone_missions_gdf['mission_type'] = drone_missions_gdf.apply(classify_mission, axis=1)

    print("Number of missions classified as High-Nadir: ",len(drone_missions_gdf[drone_missions_gdf["mission_type"] == "high-nadir"]))
    print("Number of missions classified as Low-Oblique: ",len(drone_missions_gdf[drone_missions_gdf["mission_type"] == "low-oblique"]))

    # Next, we need to create pairs of high-nadir and low-oblique missions based on year
    # Condition: needs to be collected less than 6 months apart OR in same calendar year

    # Convert date column to datetime
    drone_missions_gdf['date'] = pd.to_datetime(drone_missions_gdf['date'], errors='coerce')

    # Drop rows with unknown mission_type or missing date
    valid_missions = drone_missions_gdf[
        drone_missions_gdf['mission_type'].isin(['high-nadir', 'low-oblique']) & 
        drone_missions_gdf['date'].notna()
    ].copy()

    # Split into two groups
    high_nadir = valid_missions[valid_missions['mission_type'] == 'high-nadir'].copy()
    low_oblique = valid_missions[valid_missions['mission_type'] == 'low-oblique'].copy()

    # Create all possible pairs. This will create a Cartesian product of the two DataFrames
    paired = high_nadir.merge(
        low_oblique,
        how='cross',
        suffixes=('_hn', '_lo')
    )

    # Filter pairs by date of collection.
    def is_valid_pair(row):
        delta = abs(row['date_hn'] - row['date_lo'])
        same_year = row['date_hn'].year == row['date_lo'].year
        within_6_months = delta.days <= 183  # ~6 months
        return same_year or within_6_months

    paired_valid = paired[paired.apply(is_valid_pair, axis=1)].reset_index(drop=True)

    # This will give us a DataFrame with pairs of high-nadir and low-oblique missions
    # however, there can be multiple matches for the same high-nadir mission and vice versa.
    # TODO: Decide how to handle this, e.g., keep all pairs or have one-to-one matches.
    # For now, drop duplicates based on high-nadir mission, retaining the pair with shortest date difference

    # Calculate absolute date difference
    paired_valid['date_diff_days'] = (paired_valid['date_hn'] - paired_valid['date_lo']).abs().dt.days

    # Sort so smaller date differences come first
    paired_valid_sorted = paired_valid.sort_values('date_diff_days')

    # Drop duplicates based on high-nadir mission, keeping only closest match
    # Note: This can have the same low-oblique mission matched to multiple high-nadir missions
    paired_drone_missions_gdf = paired_valid_sorted.drop_duplicates(subset='mission_id_hn', keep='first')

    # Next, pair ground reference plots with the drone missions pairs
    # Convert survey_date to datetime, handling various formats
    def parse_survey_date(val):
        s = str(int(val)) if pd.notna(val) else ""
        if len(s) == 4:
            return pd.to_datetime(s + "-01-01")
        elif len(s) == 6:
            return pd.to_datetime(s, format="%Y%m")
        elif len(s) == 8:
            return pd.to_datetime(s, format="%Y%m%d")
        return pd.NaT

    ground_ref_plots_gdf['survey_date_parsed'] = ground_ref_plots_gdf['survey_date'].apply(parse_survey_date)

    # TODO: Confirm if it right to use the high-nadir geometry from the pair to compare with ground geometry
    paired_drone_missions_gdf = gpd.GeoDataFrame(paired_drone_missions_gdf, geometry='geometry_hn')
    paired_drone_missions_gdf.to_crs(3310, inplace=True)  # Project to meters-based CRS

    # Set drone_date to the later date of the two missions in the pair
    paired_drone_missions_gdf['drone_date'] = pd.to_datetime(paired_drone_missions_gdf[['date_hn', 'date_lo']].max(axis=1))

    # Project ground reference plots to the same CRS as the drone missions
    ground_ref_plots_gdf.to_crs(paired_drone_missions_gdf.crs, inplace=True)

    # Buffer the plots geometry by 40m
    ground_ref_plots_gdf['geometry'] = ground_ref_plots_gdf.buffer(40)

    # Find ground plots that fall within the drone geometry
    joined = gpd.sjoin(
        ground_ref_plots_gdf,
        paired_drone_missions_gdf,
        how='inner',
        predicate='within'
    )

    # Year difference should be <= 8 calendar years apart to get a valid pair
    joined['year_diff'] = (joined['drone_date'].dt.year - joined['survey_date_parsed'].dt.year).abs()
    valid_pairs = joined[joined['year_diff'] <= 8]

    ground_plot_drone_missions_matches = valid_pairs[[
        'plot_id',
        'mission_id_hn',
        'mission_id_lo',
        'date_hn',
        'date_lo',
        'survey_date_parsed',
        'year_diff',
        'geometry'  # this is the ground plot geometry
    ]].reset_index(drop=True)

    print(ground_plot_drone_missions_matches)

    # Save to file
    ground_plot_drone_missions_matches.to_file(OUTPUT_SAVE_PATH)


if __name__ == "__main__":
    # Read file with all drone missions metadata, including computed altitude values
    drone_missions_gdf = gpd.read_file(DRONE_MISSIONS_WITH_ALT)
    # Read ground reference plots
    ground_ref_plots_gdf = gpd.read_file(GROUND_REFERENCE_PLOTS)

    pair_drone_missions_and_ground_plots(drone_missions_gdf, ground_ref_plots_gdf)
