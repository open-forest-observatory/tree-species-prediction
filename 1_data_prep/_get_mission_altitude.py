import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from shapely.geometry import Point
import xml.etree.ElementTree as ET


def make_4x4_transform(rotation_str: str, translation_str: str, scale_str: str = "1"):
    """Convenience function to make a 4x4 matrix from the string format used by Metashape

    Args:
        rotation_str (str): Row major with 9 entries
        translation_str (str): 3 entries
        scale_str (str, optional): single value. Defaults to "1".

    Returns:
        np.ndarray: (4, 4) A homogenous transform mapping from cam to world
    """
    rotation_np = np.fromstring(rotation_str, sep=" ")
    rotation_np = np.reshape(rotation_np, (3, 3))

    if not np.isclose(np.linalg.det(rotation_np), 1.0, atol=1e-8, rtol=0):
        raise ValueError(
            f"Inproper rotation matrix with determinant {np.linalg.det(rotation_np)}"
        )

    translation_np = np.fromstring(translation_str, sep=" ")
    scale = float(scale_str)
    transform = np.eye(4)
    transform[:3, :3] = rotation_np * scale
    transform[:3, 3] = translation_np
    return transform


def parse_transform_metashape(camera_file):
    tree = ET.parse(camera_file)
    root = tree.getroot()
    # first level
    components = root.find("chunk").find("components")

    assert len(components) == 1
    transform = components.find("component").find("transform")
    if transform is None:
        return None

    rotation = transform.find("rotation").text
    translation = transform.find("translation").text
    scale = transform.find("scale").text

    local_to_epgs_4978_transform = make_4x4_transform(rotation, translation, scale)

    return local_to_epgs_4978_transform


def get_camera_locations(camera_file):
    """
    Parse camera locations from a Metashape XML file into a GeoDataFrame.

    Args:
        camera_file (str): Path to the Metashape .xml export file.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with camera locations as Point geometries in EPSG:4978 (ECEF),
                          with a 'label' column for camera labels.
    """
    # Load and parse the XML file
    tree = ET.parse(camera_file)
    cameras = tree.getroot().find("chunk").find("cameras")

    # Some cameras are stored in groups, so we need to flatten the structure
    ungrouped_cameras = []
    for cam_or_group in cameras:
        if cam_or_group.tag == "group":
            for cam in cam_or_group:
                ungrouped_cameras.append(cam)
        else:
            ungrouped_cameras.append(cam_or_group)

    # Collect camera-to-world transforms
    camera_locations_local = []
    camera_labels = []

    for cam in ungrouped_cameras:
        transform = cam.find("transform")
        if transform is None:
            continue

        location = np.fromstring(transform.text, sep=" ").reshape(4, 4)[:, 3:]

        camera_labels.append(cam.get("label"))
        camera_locations_local.append(location)

    camera_locations_local = np.concatenate(camera_locations_local, axis=1)

    # Get the transform from chunk to EPSG:4978
    chunk_to_epsg4978 = parse_transform_metashape(camera_file)

    if chunk_to_epsg4978 is None:
        raise ValueError("Chunk is not georeferenced")

    camera_locations_epsg4978 = chunk_to_epsg4978 @ camera_locations_local

    # Create GeoDataFrame with point geometries using the first three rows as x, y, z coordinates
    points = shapely.points(
        camera_locations_epsg4978[0, :],
        camera_locations_epsg4978[1, :],
        camera_locations_epsg4978[2, :],
    )
    gdf = gpd.GeoDataFrame({"label": camera_labels}, geometry=points, crs="EPSG:4978")
    # Transform to lat/lon
    gdf.to_crs(4326, inplace=True)
    return gdf


def main(camera_file, dtm_file, output_csv, verbose):

    # Step 1: get camera locations
    camera_set = MetashapeCameraSet(camera_file=camera_file, image_folder="")

    camera_locations = []
    for camera in camera_set.cameras:
        location = camera.local_to_epsg_4978_transform @ camera.cam_to_world_transform
        # Extract the first 3 values from the final column of the transformation matrix
        points_in_ECEF = location[:3, 3]
        camera_locations.append(Point(points_in_ECEF))

    # Step 2: Create a gdf with the camera coords in earth-centered, earth-fixed frame
    ECEF_cam_locations = gpd.GeoDataFrame(geometry=camera_locations, crs=4978)

    with rio.open(dtm_file) as dtm:
        dtm_crs = dtm.crs
        # Project to the CRS of the DTM
        DTM_crs_cam_locations = ECEF_cam_locations.to_crs(dtm_crs)

        # Step 3: Extract X, Y from projected points
        sample_coords = [(pt.x, pt.y) for pt in DTM_crs_cam_locations.geometry]

        # Step 4: Sample DTM at these coordinates with masking
        elevations = list(dtm.sample(sample_coords, masked=True))

    # Verify if at least 90% of the camera points are within the DTM
    num_total = len(elevations)
    num_valid = sum(
        not elev.mask[0] for elev in elevations
    )  # mask value is True for no data

    valid_ratio = num_valid / num_total

    print(f"Valid elevation points: {num_valid}/{num_total} ({valid_ratio:.1%})")

    if valid_ratio < 0.9:
        raise ValueError(
            "Failed. More than 10% of camera points fall outside the DTM extent."
        )

    # Step 5: Process elevations and calculate height above ground
    heights_above_ground = []
    ground_elevations = []
    camera_elevations = []

    for cam_pt, elev in zip(DTM_crs_cam_locations.geometry, elevations):
        # Skip no-data points
        if elev.mask[0] == True:
            continue
        # Index 0 because each sampled elevation is a 1 element masked array
        ground_height = elev.data[0]
        cam_height = cam_pt.z
        # Record the difference in heights as the altitude
        heights_above_ground.append(cam_height - ground_height)
        ground_elevations.append(ground_height)
        camera_elevations.append(cam_height)

    # Step 6: Compute summary statistics
    heights_np = np.array(heights_above_ground)
    ground_np = np.array(ground_elevations)
    camera_np = np.array(camera_elevations)

    cv = np.std(heights_np) / np.mean(heights_np)
    correlation = np.corrcoef(ground_np, camera_np)[
        0, 1
    ]  # Get value from the correlation matrix

    # Compute sd_photogrammetry_altitude with 5th-95th percentile clipping
    lower_bound = np.percentile(heights_np, 5)
    upper_bound = np.percentile(heights_np, 95)
    filtered_altitudes = heights_np[
        (heights_np >= lower_bound) & (heights_np <= upper_bound)
    ]
    sd_photogrammetry_altitude = np.std(filtered_altitudes)

    if verbose:
        stats = {
            "count": len(heights_np),
            "mean": np.mean(heights_np),
            "std": np.std(heights_np),
            "min": np.min(heights_np),
            "max": np.max(heights_np),
            "median": np.median(heights_np),
            "cv": cv,
            "flight_terrain_correlation_photogrammetry": correlation,
            "sd_photogrammetry_altitude": sd_photogrammetry_altitude,
        }

        print("Height above ground summary stats:")
        for k, v in stats.items():
            print(f"{k}: {v:.2f}")

    # Prepare the data as a single-row dictionary
    summary_row = {
        "mean_altitude": np.mean(heights_np),
        "cv_altitude": cv,
        "flight_terrain_correlation_photogrammetry": correlation,
        "sd_photogrammetry_altitude": sd_photogrammetry_altitude,
    }

    # Convert to DataFrame and export as CSV
    summary_df = pd.DataFrame([summary_row])

    # Ensure output directory exists and save the summary
    ensure_containing_folder(output_csv)
    summary_df.to_csv(output_csv, index=False)

    print(f"Summary exported to {output_csv}")


get_camera_locations(
    "/ofo-share/argo-data/argo-output/species_project/0001_001435_001436/output/0001_001435_001436_cameras.xml"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute height above ground for cameras and export summary statistics."
    )
    parser.add_argument(
        "--camera-file", default=EXAMPLE_CAMERAS_FILENAME, help="Path to camera file"
    )
    parser.add_argument(
        "--dtm-file", default=EXAMPLE_DTM_FILE, help="Path to DTM raster file"
    )
    parser.add_argument(
        "--output-csv",
        default="altitude_summary.csv",
        help="Path to save output CSV summary",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print summary statistics to stdout"
    )

    args = parser.parse_args()
    main(**args.__dict__)
