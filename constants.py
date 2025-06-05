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

# Inputs for first preprocessing step 1_get_mission_altitude.py
MISSION_ALTITUDES_FOLDER = Path(PREPROCESSING_FOLDER, "mission_altitudes")
MISSIONS_OUTSIDE_DTM_LIST = Path("/ofo-share/scratch-amritha/tree-species-scratch/failed_missions3.txt")
