from pathlib import Path

# Locations of installed dependencies
AUTOMATE_METASHAPE_PATH = "/ofo-share/repos-david/automate-metashape"

# This path can edited if working with a copy of the data
DATA_ROOT_FOLDER = Path("/ofo-share/species-prediction-project/")

# Inputs
# TODO this could be updated to be within the DATA_ROOT_FOLDER tree
IMAGERY_DATASETS_FOLDER = Path(
    "/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted"
)

# TODO consider renaming this to "inputs" if we want to be consistent with the NRS project
RAW_FOLDER = Path(DATA_ROOT_FOLDER, "raw")
GROUND_REFERENCE_FOLDER = Path(RAW_FOLDER, "ground-reference")

GROUND_REFERENCE_PLOTS_FILE = Path(GROUND_REFERENCE_FOLDER, "ofo_ground-reference_plots.gpkg")
GROUND_REFERENCE_TREES_FILE = Path(GROUND_REFERENCE_FOLDER, "ofo_ground-reference_trees.gpkg")

# Path to parent remote folder with all missions
ALL_MISSIONS_REMOTE_FOLDER = "js2s3:ofo-public/drone/missions_01"

# Intermediate
INTERMEDIATE_DATA_FOLDER = Path(DATA_ROOT_FOLDER, "intermediate",)
OVERLAPPING_PLOTS_FILE = Path(
    INTERMEDIATE_DATA_FOLDER, "ground_plot_drone_mission_matches.csv"
)
PHOTOGRAMMETRY_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "photogrammetry")
PREPROCESSING_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "preprocessing")

# Inputs for first preprocessing step 1_get_mission_altitude.py
MISSION_ALTITUDES_FOLDER = Path(PREPROCESSING_FOLDER, "mission_altitudes")
MISSIONS_OUTSIDE_DTM_LIST = Path(
    PREPROCESSING_FOLDER, "list_of_missions_outside_dtm.txt"
)

CHM_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "CHMs")
TREE_DETECTIONS_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "detected_trees")
SHIFTED_DRONE_TREES_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "shifted_drone_trees")
