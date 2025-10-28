import os
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon
from tqdm import tqdm

import _bootstrap
from configs.path_config import path_config


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


def main():
    plot_mission_matches = pd.read_csv(
        path_config.ground_plot_drone_mission_matches_file
    )
    # Project to meters-based CRS
    mission_meta = gpd.read_file(path_config.drone_missions_with_alt_file).to_crs(32610)
    plots_gdf = gpd.read_file(path_config.ground_reference_plots_file).to_crs(32610)

    # Buffer the plots by 100m
    plots_gdf["geometry"] = plots_gdf.geometry.buffer(100)

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
        process_mission(hn_id, "nadir", parent_folder, combined_intersection)
        process_mission(lo_id, "oblique", parent_folder, combined_intersection)

        print(f"Completed processing for plot_id: {plot_id}")


if __name__ == "__main__":
    main()
