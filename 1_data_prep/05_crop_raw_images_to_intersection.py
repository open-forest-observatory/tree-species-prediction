import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from _get_mission_altitude import compute_height_above_ground, get_camera_locations
from shapely.geometry import MultiPolygon
from tqdm import tqdm

import _bootstrap
from configs.path_config import path_config


def download_camera_and_dtm(mission_id):
    """Download the camera and dtm files to temporary local paths using rclone."""
    base_remote_path = (
        f"{path_config.all_missions_remote_folder}/{mission_id:06}/processed_02/full"
    )
    camera_file = f"{mission_id:06}_cameras.xml"
    dtm_file = f"{mission_id:06}_dtm-ptcloud.tif"

    # Remote file paths
    camera_remote = f"{base_remote_path}/{camera_file}"
    dtm_remote = f"{base_remote_path}/{dtm_file}"

    tmp_camera = tempfile.NamedTemporaryFile(suffix=".xml")
    tmp_dtm = tempfile.NamedTemporaryFile(suffix=".tif")

    camera_local = Path(tmp_camera.name)
    dtm_local = Path(tmp_dtm.name)

    # Download files to temporary paths
    subprocess.run(["rclone", "copyto", camera_remote, str(camera_local)], check=True)
    subprocess.run(["rclone", "copyto", dtm_remote, str(dtm_local)], check=True)

    return (tmp_camera, tmp_dtm)


def get_mission_geom(gdf: gpd.GeoDataFrame, mission_id: int):
    # Get the row corresponding to the mission ID
    row = gdf[gdf["mission_id"] == f"{mission_id:06d}"]
    if row.empty:
        raise ValueError(f"No geometry found for mission_id={mission_id}")

    geom = row.geometry.values[0]
    # Most geometry values are a single Polygon wrapped as MultiPolygon
    if isinstance(geom, MultiPolygon) and len(geom.geoms) == 1:
        return geom.geoms[0]
    return geom


def download_image_metadata(mission_id: str, dest_folder: Path):
    src = f"{path_config.all_missions_remote_folder}/{mission_id}/metadata-images/{mission_id}_image-metadata.gpkg"
    dest = dest_folder / f"{mission_id}_image-metadata.gpkg"
    subprocess.run(["rclone", "copyto", src, str(dest)], check=True)
    print(f"Downloaded metadata for {mission_id}")


def create_hardlinks_for_images(
    filtered_gdf, dest_folder, raw_images_root=path_config.drone_images_root
):
    for rel_path_str in filtered_gdf["image_path_ofo"]:
        rel_path = Path(
            rel_path_str
        )  # relative path like '001307/001307-02/00/001307-02_000010.JPG'
        src = raw_images_root / rel_path
        dst = dest_folder / rel_path
        if src.exists():
            dst.parent.mkdir(
                parents=True, exist_ok=True
            )  # create all parent folders if needed
            try:
                shutil.copy(src, dst)
            except FileExistsError:
                pass  # already linked, ignore
        else:
            print(f"Image not found: {src}")


def process_mission(mission_id, mission_type, parent_folder, combined_intersection):
    mission_str = f"{mission_id:06d}"
    subfolder = parent_folder / mission_type
    subfolder.mkdir(parents=True, exist_ok=True)

    # Download image metadata using rclone
    download_image_metadata(mission_str, subfolder)
    meta_path = subfolder / f"{mission_str}_image-metadata.gpkg"

    # Load metadata and filter to retain only images within the intersection
    image_metadata = gpd.read_file(meta_path)
    original_crs = image_metadata.crs
    image_metadata = image_metadata.to_crs(32610)
    filtered = image_metadata[image_metadata.geometry.within(combined_intersection)]
    filtered.to_crs(original_crs, inplace=True)

    # Save filtered metadata and create hardlinks
    filtered.to_file(meta_path, driver="GPKG")
    print(f"{mission_type}: saved {len(filtered)} filtered points")
    create_hardlinks_for_images(filtered, subfolder)

    # Compute the mean altitude above ground for the filtered images
    # Download the camera and dtm files from single-mission photogrammetry results from S3
    tmp_camera, tmp_dtm = download_camera_and_dtm(mission_id)

    # Compute the height above ground for each photogrammetry camera location
    camera_local = Path(tmp_camera.name)
    dtm_local = Path(tmp_dtm.name)

    # Extract the camera locations as estimated by photogrammetry
    cam_locations = get_camera_locations(camera_local)
    # Subtract the photogrammetry-derived DTM altitude from the cameras
    camera_elevations = compute_height_above_ground(cam_locations, dtm_local)

    # Reformat the label of the cameras to match what's in the "filtered" dataframe
    camera_elevations.label = camera_elevations.label.str.replace(
        "/data/03_input-images/", ""
    )

    # Merge, retaining only the images present in both dataframes. Note, this drops images outside
    # of the crop as well as those that didn't align during photogrammetry.
    filtered_images = pd.merge(
        filtered,
        camera_elevations,
        left_on="image_path_ofo",
        right_on="label",
        how="inner",
    )
    # Also drop images that were outside of the DTM extent
    filtered_images = filtered_images[filtered_images["valid"]]
    # Compute and return the mean altitude above ground
    mean_altitude_agl = filtered_images["altitude_agl"].mean()
    return mean_altitude_agl


def main():
    plot_mission_matches = pd.read_csv(
        path_config.ground_plot_drone_mission_matches_file
    )
    # Project to meters-based CRS
    mission_meta = gpd.read_file(path_config.drone_missions_with_alt_file).to_crs(32610)
    plots_gdf = gpd.read_file(path_config.ground_reference_plots_file).to_crs(32610)

    # Buffer the plots by 100m
    plots_gdf["geometry"] = plots_gdf.geometry.buffer(100)

    # The derived per-mission altitudes will be stored here
    computed_altitudes = []

    for _, row in tqdm(
        plot_mission_matches.iterrows(), total=len(plot_mission_matches)
    ):
        plot_id = row["plot_id"]
        hn_id = int(row["mission_id_hn"])
        lo_id = int(row["mission_id_lo"])

        # Create parent folder as plotID_nadirmissionID_obliquemissionID
        parent_folder = (
            Path(path_config.paired_image_sets_for_photogrammetry)
            / f"{plot_id:04d}_{hn_id:06d}_{lo_id:06d}"
        )
        if parent_folder.exists() and len(list(parent_folder.rglob("*"))) > 1:
            print("Skipping existing folder:", parent_folder)
            continue
        parent_folder.mkdir(parents=True, exist_ok=True)

        # Get drone mission geometry polygons
        hn_geom = get_mission_geom(mission_meta, hn_id)
        lo_geom = get_mission_geom(mission_meta, lo_id)

        # Get the corresponding buffered ground plot geometry
        row = plots_gdf[plots_gdf["plot_id"] == f"{plot_id:04d}"]
        if row.empty:
            raise ValueError(f"No plot found with ID={plot_id}")
        buffered_plot = row.geometry.values[0]

        # Get the intersection of the buffered plot with both mission geometries
        # Buffering by 0 to fix invalid geometries, if any
        combined_intersection = (
            (buffered_plot.buffer(0))
            .intersection(hn_geom.buffer(0))
            .intersection(lo_geom.buffer(0))
        )
        nadir_altitude = process_mission(
            hn_id, "nadir", parent_folder, combined_intersection
        )
        oblique_altitude = process_mission(
            lo_id, "oblique", parent_folder, combined_intersection
        )

        computed_altitudes.append(
            (plot_id, hn_id, lo_id, nadir_altitude, oblique_altitude)
        )
        print(f"Completed processing for plot_id: {plot_id}")

    # Build a dataframe of computed altitudes and save as a CSV so it can be read in the
    # photogrammetry step
    altitudes_df = pd.DataFrame(
        computed_altitudes,
        columns=[
            "plot_id",
            "nadir_mission_id",
            "oblique_mission_id",
            "nadir_mean_altitude_agl",
            "oblique_mean_altitude_agl",
        ],
    )
    altitudes_df.to_csv(path_config.drone_mission_altitudes_per_plot_file, index=False)


if __name__ == "__main__":
    main()
