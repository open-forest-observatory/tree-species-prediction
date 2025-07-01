import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from pathlib import Path
from tqdm import tqdm
import subprocess
import os

# Paths
MATCH_CSV = "/ofo-share/species-prediction-project/intermediate/ground_plot_drone_mission_matches.csv"
MISSION_META_GPKG = "/ofo-share/species-prediction-project/intermediate/preprocessing/ofo-all-missions-metadata-with-altitude.gpkg"
GROUND_PLOT_GPKG = "/ofo-share/species-prediction-project/raw/ground-reference/ofo_ground-reference_plots.gpkg"
OUTPUT_ROOT = "/ofo-share/scratch-amritha/tree-species-scratch/cropped_raw_images/"
REMOTE_STORE = "js2s3:ofo-public/drone/missions_01"
RAW_IMAGES_ROOT = "/ofo-share/catalog-data-prep/01_raw-imagery-ingestion/2_sorted"


def pad_id(mission_id: int) -> str:
    return f"{mission_id:06d}"


def unwrap_if_single_multipolygon(geom):
    if isinstance(geom, MultiPolygon) and len(geom.geoms) == 1:
        return geom.geoms[0]
    return geom


def buffer_plot_geom(geom, buffer_distance=100):
    if isinstance(geom, Polygon):
        return geom.buffer(buffer_distance)
    elif isinstance(geom, MultiPolygon):
        buffered = [g.buffer(buffer_distance) for g in geom.geoms]
        return unary_union(buffered)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")


def get_mission_geom(gdf: gpd.GeoDataFrame, mission_id: int):
    row = gdf[gdf["mission_id"] == f"{mission_id:06d}"]
    if row.empty:
        raise ValueError(f"No geometry found for mission_id={mission_id}")
    return unwrap_if_single_multipolygon(row.geometry.values[0])


def get_plot_geom(gdf: gpd.GeoDataFrame, plot_id):
    row = gdf[gdf["plot_id"] == f"{plot_id:04d}"]
    if row.empty:
        raise ValueError(f"No plot found with ID={plot_id}")
    return row.iloc[0].geometry


def download_image_metadata(mission_id: str, dest_folder: Path):
    src = f"{REMOTE_STORE}/{mission_id}/metadata-images/{mission_id}_image-metadata.gpkg"
    dest = dest_folder / f"{mission_id}_image-metadata.gpkg"
    subprocess.run(["rclone", "copyto", src, str(dest)], check=True)
    print(f"Downloaded metadata for {mission_id}")


def create_hardlinks_for_images(filtered_gdf, dest_folder, raw_images_root=Path(RAW_IMAGES_ROOT)):
    for rel_path_str in filtered_gdf["image_path_ofo"]:
        rel_path = Path(rel_path_str)  # relative path like '001307/001307-02/00/001307-02_000010.JPG'
        src = raw_images_root / rel_path
        dst = dest_folder / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)  # create all parent folders if needed
            try:
                os.link(src, dst)
            except FileExistsError:
                pass  # already linked, ignore
        else:
            print(f"Warning: Image not found: {src}")

def process_mission(mission_id, role, parent_folder, combined_intersection):
    mission_str = pad_id(mission_id)
    subfolder = parent_folder / role
    subfolder.mkdir(parents=True, exist_ok=True)

    # Step 1: Download image metadata using rclone
    download_image_metadata(mission_str, subfolder)
    meta_path = subfolder / f"{mission_str}_image-metadata.gpkg"

    # Step 2: Load metadata and filter
    gdf = gpd.read_file(meta_path)
    original_crs = gdf.crs
    gdf = gdf.to_crs(32610)
    filtered = gdf[gdf.geometry.within(combined_intersection)]
    filtered = filtered.to_crs(original_crs)

    # Step 3: Save filtered metadata and create hardlinks
    filtered.to_file(meta_path, driver="GPKG")
    print(f"{role.capitalize()}: saved {len(filtered)} filtered points")
    create_hardlinks_for_images(filtered, subfolder)


def main():
    df = pd.read_csv(MATCH_CSV)
    mission_meta = gpd.read_file(MISSION_META_GPKG).to_crs(32610)
    plots_gdf = gpd.read_file(GROUND_PLOT_GPKG).to_crs(32610)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        plot_id = row["plot_id"]
        hn_id = int(row["mission_id_hn"])
        lo_id = int(row["mission_id_lo"])
        hn_str, lo_str = pad_id(hn_id), pad_id(lo_id)

        parent_folder = Path(OUTPUT_ROOT) / f"{plot_id:04d}_{hn_str}_{lo_str}"
        parent_folder.mkdir(parents=True, exist_ok=True)

        hn_geom = get_mission_geom(mission_meta, hn_id)
        lo_geom = get_mission_geom(mission_meta, lo_id)

        raw_plot_geom = get_plot_geom(plots_gdf, plot_id)
        buffered_plot = buffer_plot_geom(raw_plot_geom, buffer_distance=100)

        combined_intersection = buffered_plot.intersection(hn_geom).intersection(lo_geom)

        process_mission(hn_id, "nadir", parent_folder, combined_intersection)
        process_mission(lo_id, "oblique", parent_folder, combined_intersection)

        print(f"Completed processing for plot_id: {plot_id}")

if __name__ == "__main__":
    main()