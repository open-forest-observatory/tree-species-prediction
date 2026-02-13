from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.mask import mask

import _bootstrap
from configs.path_config import path_config

# Path to .gpkg files containing tree crown polygons and attributes
VECTOR_DIR = path_config.drone_crowns_with_field_attributes

# Base directory to access orthomosaics
# ORTHO_BASE_DIR = path_config.photogrammetry_folder
ORTHO_BASE_DIR = Path(
    "/ofo-share/argo-data/argo-output/archive_20260202/species_project"
)

# CSV file that contains the train/val split information
SPLIT_CSV = path_config.train_val_split_file

# Path to species crosswalk file that maps species code to different lumping levels
CROSSWALK_CSV = path_config.species_class_crosswalk_file

# Base dir to save the cropped images
OUTPUT_BASE = path_config.cropped_ortho_trees

# Lumping level to use for species classes (1-4)
LUMPING_LEVEL = 2


def load_crosswalk(crosswalk_path, lumping_level):
    df = pd.read_csv(crosswalk_path)
    level_col = f"species_code_l{lumping_level}"
    primary_col = f"primary_species_l{lumping_level}"

    # Only include species where primary_species flag is True
    df_filtered = df[df[primary_col] == True]
    mapping = dict(zip(df_filtered["species_code"], df_filtered[level_col]))

    print(f"Using lumping level {lumping_level}")
    print(f"Number of classes: {len(df_filtered[level_col].dropna().unique())}")
    print(f"Total species codes mapped: {len(mapping)}")
    return mapping


def crop_trees(ortho_path, vector_path, output_base_dir, dataset_name, species_mapping):
    vector_gdf = gpd.read_file(vector_path)

    with rasterio.open(ortho_path) as src:
        if vector_gdf.crs != src.crs:
            vector_gdf = vector_gdf.to_crs(src.crs)

        vector_gdf["lumped_species"] = vector_gdf["species_code"].map(species_mapping)

        output_path = Path(output_base_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create class-level subdirectories
        for species in vector_gdf["lumped_species"].dropna().unique():
            (output_path / str(species)).mkdir(exist_ok=True)

        successful = 0
        failed = 0
        skipped = 0

        for _, row in vector_gdf.iterrows():
            species = row["lumped_species"]
            unique_id = row["unique_ID"]

            if pd.isna(species) or species == "":
                # When map() is called with a species_code that doesn't exist in the mapping dictionary (non-primary), it returns NaN.
                skipped += 1
                continue

            try:
                # Crop the ortho image using the geometry of the tree crown polygon
                # The background is set to gray (pixel value 128) to be consistent with the raw image tree crops
                out_image, _ = mask(
                    src, [row["geometry"]], crop=True, filled=True, nodata=128
                )

                # Has no pixels - edge case
                if out_image.size == 0:
                    failed += 1
                    continue

                # Take only the first 3 channels for RGB
                if src.count >= 3:
                    rgb_image = out_image[:3, :, :]
                # If there's only 1 channel (e.g. CHM), repeat it to create a 3-channel image
                elif src.count == 1:
                    rgb_image = np.repeat(out_image, 3, axis=0)
                else:
                    failed += 1
                    continue

                # Transpose from (C, H, W) to (H, W, C) for saving using PIL
                rgb_image = np.transpose(rgb_image, (1, 2, 0))

                # Normalize pixel values to 0-255 range
                if rgb_image.dtype != np.uint8:
                    rgb_image = (
                        (rgb_image - rgb_image.min())
                        / (rgb_image.max() - rgb_image.min())
                        * 255
                    ).astype(np.uint8)

                filename = f"{dataset_name}_tree{unique_id}.png"
                output_file = output_path / str(species) / filename

                Image.fromarray(rgb_image, mode="RGB").save(output_file, "PNG")
                successful += 1

            except Exception as e:
                failed += 1
                continue

        print(
            f"{dataset_name}: {successful} success, {failed} failed, {skipped} skipped"
        )
        return successful, failed, skipped


def process_all_datasets(
    lumping_level, vector_dir, ortho_base_dir, split_csv, crosswalk_csv, output_base
):

    # Load train/val split CSV
    train_val_split = pd.read_csv(split_csv)
    train_val_dict = dict(zip(train_val_split["dataset"], train_val_split["split"]))

    # Get all available datasets from the drone_crowns_with_field_attributes directory
    vector_files = list(Path(vector_dir).glob("*.gpkg"))
    all_dataset_names = set([vector_path.stem for vector_path in vector_files])

    # Identify test datasets (those not in train/val split)
    train_val_datasets = set(train_val_dict.keys())
    test_datasets = all_dataset_names - train_val_datasets

    print(f"Found {len(vector_files)} vector files")
    print(f"Train/Val datasets: {len(train_val_datasets)}")
    print(f"Test datasets: {len(test_datasets)}")

    species_mapping = load_crosswalk(crosswalk_csv, lumping_level)

    total_stats = {
        "train": {"success": 0, "failed": 0, "skipped": 0, "datasets": 0},
        "val": {"success": 0, "failed": 0, "skipped": 0, "datasets": 0},
        "test": {"success": 0, "failed": 0, "skipped": 0, "datasets": 0},
    }

    for vector_path in vector_files:
        dataset_name = vector_path.stem

        # Determine split: train, val, or test
        if dataset_name in train_val_dict:
            split = train_val_dict[dataset_name]
        elif dataset_name in test_datasets:
            split = "test"
        else:
            continue

        ortho_path = (
            Path(ortho_base_dir)
            / dataset_name
            / "output"
            / f"{dataset_name}_ortho-dsm-ptcloud.tif"
        )

        if not ortho_path.exists():
            print(f"Missing ortho for {dataset_name}, skipping")
            continue

        output_dir = Path(output_base) / split

        try:
            success, failed, skipped = crop_trees(
                str(ortho_path),
                str(vector_path),
                str(output_dir),
                dataset_name,
                species_mapping,
            )
            total_stats[split]["success"] += success
            total_stats[split]["failed"] += failed
            total_stats[split]["skipped"] += skipped
            total_stats[split]["datasets"] += 1
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    print("FINAL SUMMARY:")
    for split in ["train", "val", "test"]:
        stats = total_stats[split]
        print(
            f"{split}: {stats['datasets']} datasets, {stats['success']} trees cropped, {stats['failed']} failed, {stats['skipped']} skipped"
        )

        output_dir = Path(output_base) / split
        if output_dir.exists():
            for species_dir in sorted(output_dir.iterdir()):
                if species_dir.is_dir():
                    count = len(list(species_dir.glob("*.png")))
                    print(f"{species_dir.name}: {count}")


if __name__ == "__main__":
    print(f"Starting ortho cropping with lumping level {LUMPING_LEVEL}")
    process_all_datasets(
        lumping_level=LUMPING_LEVEL,
        vector_dir=VECTOR_DIR,
        ortho_base_dir=ORTHO_BASE_DIR,
        split_csv=SPLIT_CSV,
        crosswalk_csv=CROSSWALK_CSV,
        output_base=OUTPUT_BASE,
    )
