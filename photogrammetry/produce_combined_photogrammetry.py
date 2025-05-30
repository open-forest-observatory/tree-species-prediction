import sys
from pathlib import Path
import tempfile

# TODO consider other ways to find this location
sys.path.append("/ofo-share/repos-david/automate-metashape/python")
from metashape_workflow_functions import MetashapeWorkflow

# TODO consider whether this can be a default within MetashapeWorkflow
DEFAULT_CONFIG = Path(
    "/ofo-share/repos-david/automate-metashape/config/config-base.yml"
)
# Input and output processing paths. Could be updated.
IMAGERY_DATASETS_FOLDER = Path(
    "/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted"
)
PROJECT_FOLDER = Path("/ofo-share/repos-david/tree-species-prediction/scratch/projects")
OUTPUT_FOLDER = Path("/ofo-share/repos-david/tree-species-prediction/scratch/outputs")


def produce_combined(nadir_dataset_id, oblique_dataset_id):
    # Find the path to the imagery datasets.
    # TODO, could be updated to download data from CyVerse
    nadir_dataset_path = Path(IMAGERY_DATASETS_FOLDER, nadir_dataset_id)
    oblique_dataset_path = Path(IMAGERY_DATASETS_FOLDER, oblique_dataset_id)
    # Find the sub-folders, corresponding to sub-missions of this dataset
    nadir_sub_missions = [str(f) for f in nadir_dataset_path.glob("*") if f.is_dir()]
    oblique_sub_missions = [
        str(f) for f in oblique_dataset_path.glob("*") if f.is_dir()
    ]
    # Compute the output and project folders
    output_folder = Path(OUTPUT_FOLDER, f"{nadir_dataset_id}_{oblique_dataset_id}")
    project_folder = Path(PROJECT_FOLDER, f"{nadir_dataset_id}_{oblique_dataset_id}")
    # Build and override dict that will update the base config with run-specific information
    override_dict = {
        "photo_path": nadir_sub_missions,
        "photo_path_secondary": oblique_sub_missions,
        "output_path": str(output_folder),
        "project_path": str(project_folder),
        "run_name": f"{nadir_dataset_id}_{oblique_dataset_id}",
    }

    # Construct the workflow
    workflow = MetashapeWorkflow(DEFAULT_CONFIG, override_dict=override_dict)
    # Run the workflow
    workflow.run()


if __name__ == "__main__":
    # Test with the two valley missions
    produce_combined("000337", "000338")
