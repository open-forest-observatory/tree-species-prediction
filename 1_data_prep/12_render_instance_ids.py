import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Library imports
from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.entrypoints.render_labels import render_labels
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.visualization import show_segmentation_labels

# Ensure project root is on sys.path so _bootstrap can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
import _bootstrap
from configs.path_config import path_config

# The image is downsampled to this fraction for accelerated rendering
RENDER_IMAGE_SCALE = 0.1
# Portions of the mesh within this distance of the labels are used for rendering
MESH_BUFFER_RADIUS_METER = 20
# Cameras within this radius of the annotations are used for training
CAMERAS_BUFFER_RADIUS_METERS = 10
# Downsample target
DOWNSAMPLE_TARGET = 1

INPUT_CRS = "EPSG:26910"

# Render data from this column in the geofile to each image
# "unique_ID" column is the ID assigned to each tree-crown by the GeometricTreeCrownDetector
LABEL_COLUMN_NAME = "unique_ID"
VIS = True


if __name__ == "__main__":
    photogrammetry_folders = path_config.photogrammetry_folder.glob("*")


    for photogrammetry_folder in photogrammetry_folders:
        dataset = photogrammetry_folder.parts[-1]

        # The input labels
        LABELS_FILENAME = Path(path_config.drone_crowns_with_field_attributes, f"{dataset}.gpkg")
        # The mesh exported from Metashape
        MESH_FILENAME = Path(
            path_config.photogrammetry_folder, dataset, "outputs", f"{dataset}_model.ply"
        )
        CAMERAS_FILENAME = Path(
            path_config.photogrammetry_folder, dataset, "outputs", f"{dataset}_cameras.xml"
        )
        DTM_FILE = Path(
            path_config.photogrammetry_folder, dataset, "outputs", f"{dataset}_dtm-ptcloud.tif"
        )
        # The image folder used to create the Metashape project
        IMAGE_FOLDER = Path(path_config.raw_image_sets_folder, dataset)

        # Where to save the renders
        RENDER_FOLDER = Path(path_config.rendered_instance_ids, dataset)

        if not LABELS_FILENAME.exists():
            print(f"Skipping {dataset} - vector file not found at {LABELS_FILENAME}")
            continue

        gdf = gpd.read_file(LABELS_FILENAME)
        try:
            render_labels(
                mesh_file=MESH_FILENAME,
                cameras_file=CAMERAS_FILENAME,
                input_CRS=INPUT_CRS,
                image_folder=IMAGE_FOLDER,
                texture=gdf,
                texture_column_name=LABEL_COLUMN_NAME,
                render_savefolder=RENDER_FOLDER,
                DTM_file=DTM_FILE,
                ground_height_threshold=2,
                cameras_ROI_buffer_radius_meters=20,
                mesh_ROI_buffer_radius_meters=20,
                cast_to_uint8=False, # Save the rendered IDs as TIF files to support ID values > 255
            )

            if VIS:
                show_segmentation_labels(
                    label_folder=RENDER_FOLDER,
                    image_folder=IMAGE_FOLDER,
                    savefolder=str(RENDER_FOLDER) + "_vis",
                    num_show=10,
                    label_suffix=".tif",
                )
        except FileNotFoundError as e:
            print(
                f"skipping dataset {dataset} because of missing files. The error was the following:"
            )
            print(e)
