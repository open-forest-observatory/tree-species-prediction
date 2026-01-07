from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np

# Geograypher imports
from geograypher.entrypoints.render_labels import render_labels
from geograypher.utils.visualization import show_segmentation_labels

import _bootstrap
from configs.path_config import path_config

# The image is downsampled to this fraction for accelerated rendering
# Temporarily fixed to 1 since distortion modeling doesn't yet work on scaled images
RENDER_IMAGE_SCALE = 0.25
# Portions of the mesh within this distance of the labels are used for rendering
MESH_BUFFER_RADIUS_METER = 50
# Cameras within this radius of the annotations are used for training
CAMERAS_BUFFER_RADIUS_METERS = 80
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

# Error logging file
ERROR_LOG_FILE = Path("error_log.txt")

# Number of multiprocessing processes
N_PROCESSES = 16


def render_dataset(dataset: str) -> bool:
    """Render the tree crown IDs to each image that obseverse them

    Args:
        dataset (str): The dataset ID to render, in the <plot>_<low nadir>_<high oblique> format

    Returns:
        bool: Whether the dataset rendered without skipping or failure
    """
    # INPUTS
    # The input labels
    labels_file = Path(
        path_config.drone_crowns_with_field_attributes, f"{dataset}.gpkg"
    )
    # The paths to the photogrammetry producs
    photogrammetry_outputs_folder = Path(
        path_config.photogrammetry_folder,
        dataset,
        "output",
    )
    # The mesh exported from Metashape
    mesh_file = Path(photogrammetry_outputs_folder, f"{dataset}_mesh.ply")
    cameras_file = Path(photogrammetry_outputs_folder, f"{dataset}_cameras.xml")
    dtm_file = Path(
        photogrammetry_outputs_folder,
        f"{dataset}_dtm-ptcloud.tif",
    )

    # The folder within the docker container used for photogrammetry. The paths in the camera file
    # are with respect to this location.
    original_image_folder = Path(path_config.argo_imagery_path, dataset)
    # The image folder used to create the Metashape project. This is where the imagery is now stored.
    image_folder = Path(path_config.paired_image_sets_for_photogrammetry, dataset)

    # OUTPUTS
    # Where to save the renders
    render_output_folder = Path(path_config.rendered_instance_ids, dataset)
    # Where to save the visualizations
    vis_output_folder = Path(path_config.rendered_instance_ids_vis, dataset)

    if render_output_folder.exists():
        print(f"Skipping {dataset} - renders already exist at {render_output_folder}")
        return False

    if not labels_file.exists():
        print(f"Skipping {dataset} - vector file not found at {labels_file}")
        return False

    # Create the IDs to labels mapping
    labels = gpd.read_file(labels_file)
    label_values = labels[LABEL_COLUMN_NAME].tolist()
    # Notably, the label IDs must start at 1 because 0 is reserved for the background
    ids_to_labels = {i + 1: label for i, label in enumerate(label_values)}

    try:
        # Actually perform the rendering
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
            ROI=labels,
            cameras_ROI_buffer_radius_meters=CAMERAS_BUFFER_RADIUS_METERS,
            mesh_ROI_buffer_radius_meters=MESH_BUFFER_RADIUS_METER,
            render_image_scale=RENDER_IMAGE_SCALE,
            mesh_downsample=DOWNSAMPLE_TARGET,
            cast_to_uint8=False,  # Save the rendered IDs as TIF files to support ID values > 255
            IDs_to_labels=ids_to_labels,
        )

        if VIS:
            # Show a subset of rendered labels, colormapped
            show_segmentation_labels(
                label_folder=render_output_folder,
                image_folder=image_folder,
                savefolder=vis_output_folder,
                num_show=10,
                label_suffix=".tif",
                IDs_to_labels=ids_to_labels,
            )
    except Exception as e:
        print(
            f"skipping dataset {dataset} because of error. The error was the following:"
        )
        print(e)
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(f"Dataset {dataset} failed with error: {e}\n")
        return False

    # Everthing completed ok
    return True


if __name__ == "__main__":
    if not path_config.photogrammetry_folder.is_symlink():
        # symlink from where argo produced the photogrammetry outputs to the working file tree
        path_config.photogrammetry_folder.symlink_to(
            path_config.photogrammetry_folder_argo
        )
    # List all of the photogrammetry folders
    photogrammetry_folders = path_config.photogrammetry_folder.glob("*")

    # Compute the dataset names
    datasets = [pf.parts[-1] for pf in photogrammetry_folders]

    # Run the computation as a multiprocessing pool
    with Pool(N_PROCESSES) as p:
        # Get the list of returns
        successes = p.map(render_dataset, datasets, chunksize=1)

    # Compute which datasets failed, if any
    failed_datasets = np.array(datasets)[np.logical_not(successes)].tolist()

    if len(failed_datasets) > 0:
        print(f"The following datasets failed or were skipped: {failed_datasets}")
    else:
        print("All datasets were run and completed successfully")
