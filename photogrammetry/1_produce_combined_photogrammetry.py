import sys
from pathlib import Path

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    AUTOMATE_METASHAPE_PATH,
    IMAGERY_DATASETS_FOLDER,
    PHOTOGRAMMETRY_FOLDER,
)

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import METASHAPE_CONFIG,DERIVED_METASHAPE_CONFIGS_FOLDER, RAW_IMAGE_SETS_FOLDER

# TODO consider other ways to find this location
sys.path.append(str(Path(AUTOMATE_METASHAPE_PATH, "python")))
from metashape_workflow_functions import make_derived_yaml


def produce_combined_config(imagery_folder: Path):
    # Extract the last part of the path, which is the "<plot_id>_<nadir_id>_<oblique_id>" string
    run_name = imagery_folder.name
    # Find the path to the imagery datasets.
    nadir_dataset_path = Path(imagery_folder, "nadir")
    oblique_dataset_path = Path(imagery_folder, "oblique")
    # Find the sub-folders, corresponding to sub-missions of this dataset
    nadir_sub_missions = [str(f) for f in nadir_dataset_path.glob("*") if f.is_dir()]
    oblique_sub_missions = [
        str(f) for f in oblique_dataset_path.glob("*") if f.is_dir()
    ]

    # Create the output folders for photogrammetry outputs
    project_folder = Path(PHOTOGRAMMETRY_FOLDER, run_name)
    output_folder = Path(project_folder, "outputs")

    # Build and override dict that will update the base config with run-specific information
    override_dict = {
        "photo_path": nadir_sub_missions,
        "photo_path_secondary": oblique_sub_missions,
        "output_path": str(output_folder),
        "project_path": str(project_folder),
        "run_name": run_name,
    }

    output_config_file = Path(DERIVED_METASHAPE_CONFIGS_FOLDER, run_name + ".yml")
    # Save the derived config
    make_derived_yaml(METASHAPE_CONFIG, output_path=output_config_file, override_options=override_dict)


if __name__ == "__main__":
    # List all the imagery folders
    imagery_sets =RAW_IMAGE_SETS_FOLDER.glob("*")
    # For each folder, produce the corresponding config
    for imagery_set in imagery_sets:
        produce_combined_config(imagery_set)
