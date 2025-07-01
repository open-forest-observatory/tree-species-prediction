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

# Path to parent remote folder with all missions
ALL_MISSIONS_REMOTE_FOLDER = "js2s3:ofo-public/drone/missions_01"

# Intermediate
PHOTOGRAMMETRY_FOLDER = Path(DATA_ROOT_FOLDER, "intermediate", "photogrammetry")
PREPROCESSING_FOLDER = Path(DATA_ROOT_FOLDER, "intermediate", "preprocessing")
RAW_IMAGE_SETS_FOLDER = Path(DATA_ROOT_FOLDER, "intermediate", "raw_image_sets")

# Inputs for first preprocessing step 1_get_mission_altitude.py
MISSION_ALTITUDES_FOLDER = Path(PREPROCESSING_FOLDER, "mission_altitudes")
MISSIONS_OUTSIDE_DTM_LIST = Path(
    PREPROCESSING_FOLDER, "list_of_missions_outside_dtm.txt"
)
# Inputs for 4_pair_drone_with_ground.py
GROUND_REFERENCE_PLOTS_FILE = Path(GROUND_REFERENCE_FOLDER, "ofo_ground-reference_plots.gpkg")
DRONE_MISSIONS_WITH_ALT_FILE = Path(PREPROCESSING_FOLDER, "ofo-all-missions-metadata-with-altitude.gpkg")

GROUND_PLOT_DRONE_MISSION_MATCHES_FILE = Path(DATA_ROOT_FOLDER, "intermediate", "ground_plot_drone_mission_matches.csv")

CHM_FOLDER = Path(DATA_ROOT_FOLDER, "intermediate", "CHMs")
TREE_DETECTIONS_FOLDER = Path(DATA_ROOT_FOLDER, "intermediate", "detected_trees")

DRONE_IMAGES_ROOT = Path("/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted")
