import sys
from pathlib import Path

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    AUTOMATE_METASHAPE_PATH,
    IMAGERY_DATASETS_FOLDER,
    PHOTOGRAMMETRY_FOLDER,
)

# TODO consider other ways to find this location
sys.path.append(str(Path(AUTOMATE_METASHAPE_PATH, "python")))
from metashape_workflow_functions import MetashapeWorkflow

# The path to the config file that has the default processing parameters
DEFAULT_METASHAPE_CONFIG = Path(AUTOMATE_METASHAPE_PATH, "config", "config-base.yml")


def produce_combined(nadir_dataset_id, oblique_dataset_id):
    # Find the path to the imagery datasets.
    # TODO, could be updated to download data from JS2 Object store
    nadir_dataset_path = Path(IMAGERY_DATASETS_FOLDER, nadir_dataset_id)
    oblique_dataset_path = Path(IMAGERY_DATASETS_FOLDER, oblique_dataset_id)
    # Find the sub-folders, corresponding to sub-missions of this dataset
    nadir_sub_missions = [str(f) for f in nadir_dataset_path.glob("*") if f.is_dir()]
    oblique_sub_missions = [
        str(f) for f in oblique_dataset_path.glob("*") if f.is_dir()
    ]
    paired_id = f"{nadir_dataset_id}_{oblique_dataset_id}"

    project_folder = Path(PHOTOGRAMMETRY_FOLDER, paired_id)
    output_folder = Path(project_folder, "outputs")

    # Build and override dict that will update the base config with run-specific information
    override_dict = {
        "photo_path": nadir_sub_missions,
        "photo_path_secondary": oblique_sub_missions,
        "output_path": str(output_folder),
        "project_path": str(project_folder),
        "run_name": f"{nadir_dataset_id}_{oblique_dataset_id}",
    }

    # Construct the workflow
    workflow = MetashapeWorkflow(DEFAULT_METASHAPE_CONFIG, override_dict=override_dict)
    # Run the workflow
    workflow.run()


if __name__ == "__main__":
    # Test with the two valley missions
    produce_combined("000167", "000168")
