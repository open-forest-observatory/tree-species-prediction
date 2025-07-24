from pathlib import Path

'''from constants import (
    AUTOMATE_METASHAPE_PATH,
    DERIVED_METASHAPE_CONFIGS_FOLDER,
    METASHAPE_PYTHON_PATH,
    PHOTOGRAMMETRY_FOLDER,
    RAW_IMAGE_SETS_FOLDER,
)'''

import _bootstrap
from configs.path_config import path_config

# path resolved in _bootstrap.py
from metashape_workflow_functions import make_derived_yaml

METASHAPE_CONFIG = Path(path_config.automate_metashape_path, "config", "config-base.yml")


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
    project_folder = Path(path_config.photogrammetry_folder, run_name)
    output_folder = Path(project_folder, "outputs")

    # Build and override dict that will update the base config with run-specific information
    # Also, only generate the DSM-ptcloud orthomosaic. Note, we only need the DTM and mesh-based
    # DSM for downstream experiments, but the ptcloud-based DEM must be computed for building the
    # orthomosaic
    override_dict = {
        "photo_path": nadir_sub_missions,
        "photo_path_secondary": oblique_sub_missions,
        "output_path": str(output_folder),
        "project_path": str(project_folder),
        "run_name": run_name,
        "buildOrthomosaic": {"surface": ["DSM-ptcloud"]},
        "buildPointCloud": {"remove_after_export": True},
    }

    output_config_file = Path(path_config.derived_metashape_configs_folder, run_name + ".yml")
    # Save the derived config
    make_derived_yaml(
        METASHAPE_CONFIG, output_path=output_config_file, override_options=override_dict
    )


def make_photogrammetry_run_scripts(n_chunks=4):
    # List all configs
    derived_configs = sorted(path_config.derived_metashape_configs_folder.glob("*yml"))
    # TODO consider shuffling the files in case there's a structure to which datasets are large
    # The path to the metashape runner script
    metashape_script_path = str(
        Path(path_config.automate_metashape_path, "python", "metashape_workflow.py")
    )
    # Create a string that contains the python run command (one for each mission-pair to process)
    run_strings = [
        " ".join(
            [
                path_config.metashape_python_path,
                metashape_script_path,
                "--config_file",
                str(derived_config),
            ]
        )
        + "\n"
        for derived_config in derived_configs
    ]

    # Determine how many files to run per machine
    n_files_per_chunk = (len(run_strings)) / n_chunks
    # Calculate splits so that the number of lines per chunk is at most one different
    splits = [int(round(i * n_files_per_chunk)) for i in range(n_chunks + 1)]

    # Write out one file per chunk
    for i in range(n_chunks):
        # Create the named output file
        with open(
            Path(path_config.derived_metashape_configs_folder, f"run_script_{i:02}.sh"), "w"
        ) as output_file_h:
            # Get the corresponding lines and write them out
            chunk_run_strings = run_strings[splits[i] : splits[i + 1]]
            output_file_h.writelines(chunk_run_strings)


if __name__ == "__main__":
    # List all the imagery folders
    imagery_sets = path_config.raw_image_sets_folder.glob("*")
    # For each folder, produce the corresponding config
    for imagery_set in imagery_sets:
        produce_combined_config(imagery_set)

    # Create scripts that can be run on multiple machines to sequentially run projects
    make_photogrammetry_run_scripts()
