from pathlib import Path

# path resolved in _bootstrap.py
from metashape_workflow_functions import make_derived_yaml

import _bootstrap
from configs.path_config import path_config

METASHAPE_CONFIG = Path(
    path_config.automate_metashape_path, "config", "config-base.yml"
)


def produce_combined_config(imagery_folder: Path):
    # Extract the last part of the path, which is the "<plot_id>_<nadir_id>_<oblique_id>" string
    run_name = imagery_folder.name

    try:
        _, nadir_id, oblique_id = run_name.split("_")
    except ValueError:
        return
    # Find the path to the imagery datasets.
    # Note that we could skip the step of computing oblique and nadir folders and just use one glob
    # specifying that folders are nested three levels deep, but this is a little more robust to
    # spurious files/folders
    nadir_dataset_path = Path(imagery_folder, "nadir", nadir_id)
    oblique_dataset_path = Path(imagery_folder, "oblique", oblique_id)
    # Find the sub-folders, corresponding to sub-missions of this dataset, for both oblique and
    # nadir images
    sub_missions = list(nadir_dataset_path.glob("*")) + list(oblique_dataset_path.glob("*"))

    # When we run photogrammetry it's going to be within docker and the data will be mounted in a
    # volume. This will change the paths compared to what's on /ofo-share. This updates the input
    # folders so they are appropriate for docker.
    sub_missions = [
        str(
            Path(
                path_config.argo_imagery_path,
                f.relative_to(path_config.paired_image_sets_for_photogrammetry),
            )
        )
        for f in sub_missions
        if f.is_dir()
    ]

    # Build and override dict that will update the base config with the location of the input images.
    # Also, only generate the DSM-ptcloud orthomosaic. Note, we only need the DTM and mesh-based
    # DSM for downstream experiments, but the ptcloud-based DEM must be computed for building the
    # orthomosaic
    # Finally, the point clouds are removed from the project files to save space.

    override_dict = {
        "photo_path": sub_missions,
        "buildOrthomosaic": {"surface": ["DSM-ptcloud"]},
        "buildPointCloud": {"remove_after_export": True},
    }
    # Where to save the config
    output_config_file = Path(
        path_config.derived_metashape_configs_folder, run_name + ".yml"
    )
    # Save the derived config
    make_derived_yaml(
        METASHAPE_CONFIG, output_path=output_config_file, override_options=override_dict
    )


if __name__ == "__main__":
    # List all the imagery folders
    imagery_sets = list(path_config.paired_image_sets_for_photogrammetry.glob("*"))
    # For each folder, produce the corresponding config
    for imagery_set in imagery_sets:
        produce_combined_config(imagery_set)
