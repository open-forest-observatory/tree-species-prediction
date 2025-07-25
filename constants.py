from pathlib import Path

# Locations of installed dependencies
AUTOMATE_METASHAPE_PATH = "/ofo-share/repos-david/automate-metashape"
METASHAPE_PYTHON_PATH = "/home/exouser/miniconda3/envs/meta/bin/python"

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
RAW_IMAGE_SETS_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "raw_image_sets")

DERIVED_METASHAPE_CONFIGS_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "metashape_configs")

# Inputs for first preprocessing step 1_get_mission_altitude.py
MISSION_ALTITUDES_FOLDER = Path(PREPROCESSING_FOLDER, "mission_altitudes")
MISSIONS_OUTSIDE_DTM_LIST = Path(
    PREPROCESSING_FOLDER, "list_of_missions_outside_dtm.txt"
)
# Inputs for 4_pair_drone_with_ground.py
GROUND_REFERENCE_PLOTS_FILE = Path(GROUND_REFERENCE_FOLDER, "ofo_ground-reference_plots.gpkg")
ALL_MISSIONS_METADATA_FILE = Path("ofo-share/catalog-data-prep/01_raw-imagery-ingestion/metadata/3_final/ofo-all-missions-metadata.gpkg")
DRONE_MISSIONS_WITH_ALT_FILE = Path(PREPROCESSING_FOLDER, "ofo-all-missions-metadata-with-altitude.gpkg")

GROUND_PLOT_DRONE_MISSION_MATCHES_FILE = Path(INTERMEDIATE_DATA_FOLDER , "ground_plot_drone_mission_matches.csv")

DRONE_IMAGES_ROOT = Path("/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted")
CHM_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "CHMs")
TREE_DETECTIONS_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "detected_trees")
SHIFTED_FIELD_TREES_FOLDER = Path(INTERMEDIATE_DATA_FOLDER, "shifted_field_trees")

DRONE_CROWNS_WITH_FIELD_ATTRIBUTES = Path(INTERMEDIATE_DATA_FOLDER, "drone_crowns_with_field_attributes")

# Output of 6_determine_species_classes.py
SPECIES_CLASS_CROSSWALK_FILE = Path(INTERMEDIATE_DATA_FOLDER, "species_class_crosswalk.csv")
