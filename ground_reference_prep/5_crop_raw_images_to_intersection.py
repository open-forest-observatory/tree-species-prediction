import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from pathlib import Path
import subprocess
import zipfile

# Paths
MATCH_CSV = "/ofo-share/species-prediction-project/intermediate/ground_plot_drone_mission_matches.csv"
MISSION_META_GPKG = "/ofo-share/species-prediction-project/intermediate/preprocessing/ofo-all-missions-metadata-with-altitude.gpkg"
GROUND_PLOT_GPKG = "/ofo-share/species-prediction-project/raw/ground-reference/ofo_ground-reference_plots.gpkg"
OUTPUT_ROOT = Path("/ofo-share/scratch-amritha/tree-species-scratch/cropped_raw_images/")
REMOTE_STORE = "js2s3:ofo-public/drone/missions_01"


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


def download_rclone(mission_id: int, dest_folder: Path):
    mid = pad_id(mission_id)
    base_path = f"{REMOTE_STORE}/{mid}"
    files = [
        f"{base_path}/metadata-images/{mid}_image-metadata.gpkg",
        f"{base_path}/images/{mid}_images.zip"
    ]
    for file in files:
        dest_path = dest_folder / Path(file).name
        subprocess.run(["rclone", "copyto", file, str(dest_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("Download completed.")


def unzip_images(zip_path: Path, extract_to: Path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(extract_to)
    zip_ref.close()


def get_mission_geom(gdf: gpd.GeoDataFrame, mission_id: int):
    row = gdf[gdf["mission_id"] == f"{mission_id:06d}"]
    if row.empty:
        raise ValueError(f"No geometry found for mission_id={mission_id}")
    return unwrap_if_single_multipolygon(row.geometry.values[0])


def get_plot_geom(gdf: gpd.GeoDataFrame, plot_id):
    row = gdf[gdf["plot_id"] == f"{plot_id:04d}"]
    if row.empty:
        raise ValueError(f"No plot found with ID={plot_id}")
    return row.geometry.values[0]


def remove_unfiltered_images(metadata_gdf, folder: Path):
    sub_mission_ids = metadata_gdf["sub_mission_id"].unique()
    image_ids = metadata_gdf["image_id"].astype(str).str.zfill(6).tolist()

    for sub_mission_id in sub_mission_ids:
        image_folder = folder / sub_mission_id
        if not image_folder.exists():
            print(f"Warning: Folder not found: {image_folder}")
            continue

        keep_filenames = {f"{sub_mission_id}_{img_id}.JPG" for img_id in image_ids}

        for file in image_folder.iterdir():
            if file.suffix.upper() == ".JPG" and file.name not in keep_filenames:
                file.unlink()


def process_mission(mission_id: int, role: str, parent_folder: Path, combined_intersection):
    mission_str = pad_id(mission_id)
    subfolder = parent_folder / role
    subfolder.mkdir(parents=True, exist_ok=True)

    # Download and unzip
    download_rclone(mission_id, subfolder)
    unzip_images(subfolder / f"{mission_str}_images.zip", subfolder)

    # Filter metadata
    meta_path = subfolder / f"{mission_str}_image-metadata.gpkg"
    gdf = gpd.read_file(meta_path)
    filtered = gdf[gdf.geometry.within(combined_intersection)]
    filtered.to_file(meta_path, driver="GPKG")
    print(f"{role.capitalize()}: saved {len(filtered)} filtered points to {meta_path.name}")

    remove_unfiltered_images(filtered, subfolder)


def main():
    df = pd.read_csv(MATCH_CSV)
    first_row = df.iloc[0]
    plot_id = first_row["plot_id"]
    hn_id = int(first_row["mission_id_hn"])
    lo_id = int(first_row["mission_id_lo"])
    hn_str, lo_str = pad_id(hn_id), pad_id(lo_id)

    parent_folder = OUTPUT_ROOT / f"{plot_id}_{hn_str}_{lo_str}"
    parent_folder.mkdir(parents=True, exist_ok=True)

    mission_meta = gpd.read_file(MISSION_META_GPKG)
    hn_geom = get_mission_geom(mission_meta, hn_id)
    lo_geom = get_mission_geom(mission_meta, lo_id)

    plots_gdf = gpd.read_file(GROUND_PLOT_GPKG)
    raw_plot_geom = get_plot_geom(plots_gdf, plot_id)
    buffered_plot = buffer_plot_geom(raw_plot_geom, buffer_distance=100)

    combined_intersection = buffered_plot.intersection(hn_geom).intersection(lo_geom)

    process_mission(hn_id, "nadir", parent_folder, combined_intersection)
    process_mission(lo_id, "oblique", parent_folder, combined_intersection)

    print("Processing complete.")


if __name__ == "__main__":
    main()