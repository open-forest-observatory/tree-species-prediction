import sys
from pathlib import Path
from math import ceil

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    AUTOMATE_METASHAPE_PATH,
    METASHAPE_PYTHON_PATH,
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

    _, nadir_id, oblique_id = run_name.split("_")
    # Find the path to the imagery datasets.
    nadir_dataset_path = Path(imagery_folder, "nadir", nadir_id)
    oblique_dataset_path = Path(imagery_folder, "oblique", oblique_id)
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

def make_photogrammetry_run_scripts(n_chunks=4):
    derived_configs = list(DERIVED_METASHAPE_CONFIGS_FOLDER.glob("*yml"))
    metashape_script_path = str(Path(AUTOMATE_METASHAPE_PATH, "python", "metashape_workflow.py"))
    run_strings = [" ".join([METASHAPE_PYTHON_PATH, metashape_script_path, "--config_file", str(derived_config)]) + "\n" for derived_config in derived_configs]

    n_files_per_chunk = int(ceil(len(run_strings) / n_chunks))

    for i in range(n_chunks):
        output_file = Path(DERIVED_METASHAPE_CONFIGS_FOLDER, f"run_script_{i:02}.sh")
        with open(output_file, "w") as output_file_h:
            chunk_run_strings = run_strings[i * n_files_per_chunk: (i+1) * n_files_per_chunk]
            output_file_h.writelines(chunk_run_strings)

if __name__ == "__main__":
    # List all the imagery folders
    imagery_sets = RAW_IMAGE_SETS_FOLDER.glob("*")
    # For each folder, produce the corresponding config
    for imagery_set in imagery_sets:
        produce_combined_config(imagery_set)

    make_photogrammetry_run_scripts()
