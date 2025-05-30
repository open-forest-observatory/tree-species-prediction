import sys
from pathlib import Path
import tempfile

sys.path.append("/ofo-share/repos-david/automate-metashape/python")
from utils import make_derived_yaml
from metashape_workflow_functions import MetashapeWorkflow


DEFAULT_CONFIG = Path(
    "/ofo-share/repos-david/automate-metashape/config/config-base.yml"
)
IMAGERY_DATASETS_FOLDER = Path(
    "/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted"
)
PROJECT_FOLDER = Path("/ofo-share/repos-david/tree-species-prediction/scratch/projects")
OUTPUT_FOLDER = Path("/ofo-share/repos-david/tree-species-prediction/scratch/outputs")


def produce_combined(nadir_dataset_id, oblique_dataset_id):
    nadir_dataset_path = Path(IMAGERY_DATASETS_FOLDER, nadir_dataset_id)
    oblique_dataset_path = Path(IMAGERY_DATASETS_FOLDER, oblique_dataset_id)

    nadir_sub_missions = [str(f) for f in nadir_dataset_path.glob("*") if f.is_dir()]
    oblique_sub_missions = [
        str(f) for f in oblique_dataset_path.glob("*") if f.is_dir()
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        derived_config_file = Path(
            temp_dir, f"{nadir_dataset_id}_{oblique_dataset_id}.yaml"
        )
        output_folder = Path(OUTPUT_FOLDER, f"{nadir_dataset_id}_{oblique_dataset_id}")
        project_folder = Path(PROJECT_FOLDER, "project_dir")
        override_dict = {
            "photo_path": nadir_sub_missions,
            "photo_path_secondary": oblique_sub_missions,
            "output_path": str(output_folder),
            "project_path": str(project_folder),
        }

        make_derived_yaml(DEFAULT_CONFIG, derived_config_file, override_dict)

        # Construct the workflow
        workflow = MetashapeWorkflow(derived_config_file, override_dict={})
        # Run the workflow
        workflow.run()


if __name__ == "__main__":
    produce_combined("000339", "000342")
