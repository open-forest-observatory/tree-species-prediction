import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Library imports
from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.entrypoints.render_labels import render_labels
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.visualization import show_segmentation_labels

import _bootstrap
from configs.path_config import path_config

# The image is downsampled to this fraction for accelerated rendering
RENDER_IMAGE_SCALE = 0.1
# Portions of the mesh within this distance of the labels are used for rendering
MESH_BUFFER_RADIUS_METER = 20
# Cameras within this radius of the annotations are used for training
CAMERAS_BUFFER_RADIUS_METERS = 20
# Downsample target
DOWNSAMPLE_TARGET = 1
# Set points under this height to ground
GROUND_HEIGHT_THRESHOLD = 2

# CRS inferred from automate-metashape config that generated the product, metadata.xml, or sibling rasters
INPUT_CRS = "EPSG:26910"

# Render data from this column in the geofile to each image
# "unique_ID" column is the ID assigned to each tree-crown by the GeometricTreeCrownDetector
LABEL_COLUMN_NAME = "unique_ID"
VIS = True

if __name__ == "__main__":
    if not path_config.photogrammetry_folder.is_symlink():
        # symlink from where argo produced the photogrammetry outputs to the working file tree
        path_config.photogrammetry_folder.symlink_to(
            path_config.photogrammetry_folder_argo
        )
    photogrammetry_folders = path_config.photogrammetry_folder.glob("*")

    for photogrammetry_folder in photogrammetry_folders:
        dataset = photogrammetry_folder.parts[-1]

        # INPUTS
        # The input labels
        labels_file = Path(path_config.drone_crowns_with_field_attributes, f"{dataset}.gpkg")
        # The mesh exported from Metashape
        mesh_file = Path(
            path_config.photogrammetry_folder, dataset, "output", f"{dataset}_mesh.ply"
        )
        cameras_file = Path(
            path_config.photogrammetry_folder, dataset, "output", f"{dataset}_cameras.xml"
        )
        dtm_file = Path(
            path_config.photogrammetry_folder, dataset, "output", f"{dataset}_dtm-ptcloud.tif"
        )
        # The image folder used to create the Metashape project
        image_folder = Path(path_config.raw_image_sets_folder, dataset)
        original_image_folder = Path("/data/argo-input/datasets", dataset)

        # OUTPUTS
        # Where to save the renders
        render_output_folder = path_config.rendered_instance_ids / dataset
        # Where to save the visualizations
        vis_output_folder = path_config.rendered_instance_ids_vis / dataset

        if not labels_file.exists():
            print(f"Skipping {dataset} - vector file not found at {labels_file}")
            continue

        try:
            render_labels(
                mesh_file=mesh_file,
                cameras_file=cameras_file,
                mesh_CRS=INPUT_CRS,
                image_folder=image_folder,
                original_image_folder=original_image_folder,
                texture=labels_file,
                texture_column_name=LABEL_COLUMN_NAME,
                render_savefolder=render_output_folder,
                DTM_file=dtm_file,
                ground_height_threshold=GROUND_HEIGHT_THRESHOLD,
                cameras_ROI_buffer_radius_meters=CAMERAS_BUFFER_RADIUS_METERS,
                mesh_ROI_buffer_radius_meters=MESH_BUFFER_RADIUS_METER,
                render_image_scale=RENDER_IMAGE_SCALE,
                mesh_downsample=DOWNSAMPLE_TARGET,
                cast_to_uint8=False,  # Save the rendered IDs as TIF files to support ID values > 255
            )

            if VIS:
                show_segmentation_labels(
                    label_folder=render_output_folder,
                    image_folder=image_folder,
                    savefolder=vis_output_folder,
                    num_show=10,
                    label_suffix=".tif",
                )
        except FileNotFoundError as e:
            print(
                f"skipping dataset {dataset} because of missing files. The error was the following:"
            )
            print(e)
