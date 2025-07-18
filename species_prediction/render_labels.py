import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Library imports
from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.entrypoints.render_labels import render_labels
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.visualization import show_segmentation_labels

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    DRONE_CROWNS_WITH_FIELD_ATTRIBUTES,
    PHOTOGRAMMETRY_FOLDER,
    RAW_IMAGE_SETS_FOLDER,
    RENDERED_LABELS,
    SPECIES_CLASS_CROSSWALK_FILE,
)

# The image is downsampled to this fraction for accelerated rendering
RENDER_IMAGE_SCALE = 1
# Portions of the mesh within this distance of the labels are used for rendering
MESH_BUFFER_RADIUS_METER = 20
# Cameras within this radius of the annotations are used for training
CAMERAS_BUFFER_RADIUS_METERS = 10
# Downsample target
DOWNSAMPLE_TARGET = 1

INPUT_CRS = "EPSG:26910"

# Render data from this column in the geofile to each image
LABEL_COLUMN_NAME = "species_code"
VIS = True


if __name__ == "__main__":
    photogrammetry_folders = PHOTOGRAMMETRY_FOLDER.glob("*")

    # Set up a dictionary to aggregate the species
    species_remapping = pd.read_csv(SPECIES_CLASS_CROSSWALK_FILE)
    species_remapping_dict = {
        k: v
        for (k, v) in zip(
            species_remapping.species_code.tolist(),
            species_remapping.species_code_l4.tolist(),
        )
    }

    for photogrammetry_folder in photogrammetry_folders:
        dataset = photogrammetry_folder.parts[-1]

        # The input labels
        LABELS_FILENAME = Path(DRONE_CROWNS_WITH_FIELD_ATTRIBUTES, f"{dataset}.gpkg")
        # The mesh exported from Metashape
        MESH_FILENAME = Path(
            PHOTOGRAMMETRY_FOLDER, dataset, "outputs", f"{dataset}_model.ply"
        )
        CAMERAS_FILENAME = Path(
            PHOTOGRAMMETRY_FOLDER, dataset, "outputs", f"{dataset}_cameras.xml"
        )
        DTM_FILE = Path(
            PHOTOGRAMMETRY_FOLDER, dataset, "outputs", f"{dataset}_dtm-ptcloud.tif"
        )
        # The image folder used to create the Metashape project
        IMAGE_FOLDER = Path(RAW_IMAGE_SETS_FOLDER, dataset)

        # Where to save the renders
        RENDER_FOLDER = Path(RENDERED_LABELS, dataset)

        gdf = gpd.read_file(LABELS_FILENAME)
        gdf.species_code.replace(species_remapping_dict)
        # Drop rows with None values
        gdf = gdf.loc[~gdf[LABEL_COLUMN_NAME].isna()]

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
            )

            if VIS:
                show_segmentation_labels(
                    label_folder=RENDER_FOLDER,
                    image_folder=IMAGE_FOLDER,
                    savefolder=str(RENDER_FOLDER) + "_vis",
                    num_show=10,
                    label_suffix=".png",
                )
        except FileNotFoundError as e:
            print(
                f"skipping dataset {dataset} because of missing files. The error was the following:"
            )
            print(e)
