from pathlib import Path
from dataclasses import dataclass, fields
from typing import get_type_hints
import argparse

from utils.config_utils import parse_config_args

@dataclass
class PathConfig:
    # NOTE: When adding config attrs, ensure type hinting is used as shown below
    # Locations of installed dependencies
    automate_metashape_path: Path = Path("/ofo-share/repos-david/automate-metashape")
    metashape_python_path: Path = Path("/home/exouser/miniconda3/envs/meta/bin/python")
    
    # This path can edited if working with a copy of the data
    data_root_folder: Path = Path("/ofo-share/project-data/species-prediction-project/")
    
    # Inputs
    # TODO this could be updated to be within the DATA_ROOT_FOLDER tree
    imagery_datasets_folder: Path = Path(
        "/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted"
    )
    
    # TODO consider renaming this to "inputs" if we want to be consistent with the NRS project
    raw_folder: Path = Path(data_root_folder, "raw")
    ground_reference_folder: Path = Path(raw_folder, "ground-reference")

    ground_reference_plots_file: Path = Path(
        ground_reference_folder, "ofo_ground-reference_plots.gpkg"
    )
    ground_reference_trees_file: Path = Path(
        ground_reference_folder, "ofo_ground-reference_trees.gpkg"
    )

    # Path to parent remote folder with all missions
    all_missions_remote_folder: str = "js2s3:ofo-public/drone/missions_01"
    
    # Intermediate
    intermediate_data_folder: Path = Path(data_root_folder, "intermediate")
    overlapping_plots_file: Path = Path(
        intermediate_data_folder, "ground_plot_drone_mission_matches.csv"
    )
    photogrammetry_folder: Path = Path(intermediate_data_folder, "photogrammetry")
    preprocessing_folder: Path = Path(intermediate_data_folder, "preprocessing")
    raw_image_sets_folder: Path = Path(intermediate_data_folder, "raw_image_sets")

    derived_metashape_configs_folder: Path = Path(
        intermediate_data_folder, "metashape_configs"
    )
    
    # Inputs for first preprocessing step 1_get_mission_altitude.py
    mission_altitudes_folder: Path = Path(preprocessing_folder, "mission_altitudes")
    missions_outside_dtm_list: Path = Path(
        preprocessing_folder, "list_of_missions_outside_dtm.txt"
    )

    # Inputs for 4_pair_drone_with_ground.py
    drone_missions_with_alt_file: Path = Path(
        preprocessing_folder, "ofo-all-missions-metadata-with-altitude.gpkg"
    )

    all_missions_metadata_file: Path = Path("ofo-share/catalog-data-prep/01_raw-imagery-ingestion/metadata/3_final/ofo-all-missions-metadata.gpkg")

    ground_plot_drone_mission_matches_file: Path = Path(
        intermediate_data_folder, "ground_plot_drone_mission_matches.csv"
    )

    hdbscan_clustered_plots: Path = Path(intermediate_data_folder, "hdbscan_clustered_plots.gpkg")
    train_test_split_file: Path = Path(intermediate_data_folder, "train_test_split.csv")
    
    drone_images_root: Path = Path(
        "/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted"
    )
    chm_folder: Path = Path(intermediate_data_folder, "CHMs")
    tree_detections_folder: Path = Path(
        intermediate_data_folder, "detected_trees"
    )
    shifted_field_trees_folder: Path = Path(
        intermediate_data_folder, "shifted_field_trees"
    )

    drone_crowns_with_field_attributes: Path = Path(
        intermediate_data_folder, "drone_crowns_with_field_attributes"
    )

    # Output of 6_determine_species_classes.py
    species_class_crosswalk_file: Path = Path(
        intermediate_data_folder, "species_class_crosswalk.csv"
    )

    # rendered instance IDs (.tif) files (masks of tree ids corresponding to raw images)
    rendered_instance_ids_path: Path = intermediate_data_folder / 'rendered_instance_ids' / 'renders'

    # output of 12_tree_crops.py
    # cropped trees using raw images and tif file masks of tree id labels (rendered_instance_ids_path)
    cropped_tree_training_images: Path = intermediate_data_folder / 'cropped_trees_v1'

    # plant clef 2024 pretrained model as a base
    pretrained_model_path: Path = intermediate_data_folder / 'pretrained_models' / 'model_best_vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.pth.tar'

    # output of model trainings
    # temporarily relative path until testing complete
    training_ckpt_dir: Path = Path("2_training/ckpts")

    # cache of static image transformations (resizing to input dim)
    # kept local to not interfere with others' runs
    static_transformed_images_cache_dir: Path = Path("cache")

    def __setattr__(self, name, value):
        """Type enforcement if config variables are overridden"""
        hints = get_type_hints(self.__class__) # get types of class attributes
        if name in hints:
            expected = hints[name] # get expected type of attr being assigned
            if not isinstance(value, expected): # if type assigning mismatches type of attr,
                try:
                    value = expected(value) # try to typecast value to attr's type
                except Exception as e:
                    raise TypeError(
                        f"Cannot cast `{name}` to {expected.__name__!r}: {e}"
                    )
        super().__setattr__(name, value) # if above checks pass, assign value to attr

path_config, path_args = parse_config_args(PathConfig)
