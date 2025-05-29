import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from shapely.geometry import Point
from geograypher.cameras import MetashapeCameraSet
from geograypher.utils.files import ensure_containing_folder
from geograypher.constants import (
    EXAMPLE_CAMERAS_FILENAME,
    EXAMPLE_DTM_FILE,
)

def main(camera_file, dtm_file, output_csv):

    # Step 1: get camera locations
    camera_set = MetashapeCameraSet(camera_file=camera_file, image_folder="")

    camera_locations = []
    for camera in camera_set.cameras:
        location = (camera.local_to_epsg_4978_transform @ camera.cam_to_world_transform)
        # Extract the first 3 values from the final column of the transformation matrix
        points_in_ECEF = location[:3, 3]
        camera_locations.append(Point(points_in_ECEF))

    # Step 2: Create a gdf with the camera coords in earth-centered, earth-fixed frame
    ECEF_cam_locations = gpd.GeoDataFrame(
        geometry=camera_locations,
        crs=4978
    )

    with rio.open(dtm_file) as src:
        dtm_crs = src.crs
        # Project to the CRS of the DTM
        DTM_crs_cam_locations = ECEF_cam_locations.to_crs(dtm_crs)

        # Step 3: Extract X, Y from projected points
        sample_coords = [(pt.x, pt.y) for pt in DTM_crs_cam_locations.geometry]

        # Step 4: Sample DTM at these coordinates with masking
        elevations = list(src.sample(sample_coords, masked=True))

    # Verify if at least 90% of the camera points are within the DTM
    num_total = len(elevations)
    num_valid = sum(not elev.mask[0] for elev in elevations)  # mask value is True for no data

    valid_ratio = num_valid / num_total

    print(f"Valid elevation points: {num_valid}/{num_total} ({valid_ratio:.1%})")

    if valid_ratio < 0.9:
        raise ValueError("Failed. More than 10% of camera points fall outside the DTM extent.")


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
    correlation = np.corrcoef(ground_np, camera_np)[0, 1]  # Get value from the correlation matrix

    if args.verbose:
        stats = {
            "count": len(heights_np),
            "mean": np.mean(heights_np),
            "std": np.std(heights_np),
            "min": np.min(heights_np),
            "max": np.max(heights_np),
            "median": np.median(heights_np),
            "cv": cv,
            "corr_ground_vs_camera_elevation": correlation
        }

        print("Height above ground summary stats:")
        for k, v in stats.items():
            print(f"{k}: {v:.2f}")

    # Prepare the data as a single-row dictionary
    summary_row = {
        "mean_height_above_ground": np.mean(heights_np),
        "cv_height_above_ground": cv,
        "corr_ground_vs_camera_elevation": correlation
    }

    # Convert to DataFrame and export as CSV
    summary_df = pd.DataFrame([summary_row])

    # Ensure output directory exists and save the summary
    ensure_containing_folder(output_csv)
    summary_df.to_csv(output_csv, index=False)

    print(f"Summary exported to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute height above ground for cameras and export summary statistics.")
    parser.add_argument("--camera-file", default=EXAMPLE_CAMERAS_FILENAME, help="Path to camera file")
    parser.add_argument("--dtm-file", default=EXAMPLE_DTM_FILE, help="Path to DTM raster file")
    parser.add_argument("--output-csv", default="altitude_summary.csv", help="Path to save output CSV summary")
    parser.add_argument("--verbose", action="store_true", help="Print summary statistics to stdout")

    args = parser.parse_args()
    main(args.camera_file, args.dtm_file, args.output_csv)