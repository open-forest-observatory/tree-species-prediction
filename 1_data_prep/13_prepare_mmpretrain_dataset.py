import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import _bootstrap
from configs.path_config import path_config

# Configuration parameters
DATA_FOLDER_PATH = path_config.cropped_tree_training_images
OUTPUT_PATH = path_config.mmpretrain_dataset_folder
SPLIT_FILE_PATH = path_config.train_val_split_file
SPECIES_LEVEL = "l4"  # Class lumping level options: 'l1', 'l2', 'l3', 'l4'
FLIGHT_TYPE = "both"  # Options: 'nadir', 'oblique', 'both'
EXCLUDE_SPECIES = None # ["NODE3", "ARME"]  for L3

# If the cropped tree images have moved from their original location at the point of creation,
# image_path in the metadata files will need to be updated to DATA_FOLDER_PATH. Else, set to None.
REPLACE_PATH = True
OLD_BASE_PATH = "/ofo-share/repos/david/tree-species-prediction/scratch/cropped_trees"


def load_split_from_file(split_file_path):
    split_df = pd.read_csv(split_file_path)
    train_datasets = split_df[split_df['split'] == 'train']['dataset'].tolist()
    val_datasets = split_df[split_df['split'] == 'val']['dataset'].tolist()
    
    print(f"Loaded from split file:")
    print(f"  Train datasets: {len(train_datasets)}")
    print(f"  Val datasets: {len(val_datasets)}")

    return train_datasets, val_datasets


def get_all_dataset_names(labelled_path):
    all_datasets = set()
    for item in labelled_path.iterdir():
        if item.is_dir():
            all_datasets.add(item.name)
    return all_datasets


def fix_image_path(path_str, replace_path, old_base_path):
    """
    Fix image paths by replacing old base paths with the correct one.
    
    Args:
        path_str: Original path from metadata
        replace_path: New base path to use for the labelled directory
        old_base_path: Old base path to be replaced

    Returns:
        Updated path string
    """
    original_path = Path(path_str)

    # Extract the relative path after the old base
    relative_path = original_path.relative_to(old_base_path)
    # Construct new path with replace_path
    new_path = Path(replace_path) / relative_path
    return str(new_path)


def create_mmpretrain_structure(
    data_folder_path, 
    species_level, 
    split_file_path, 
    output_path, 
    flight_type="both", 
    exclude_species=None,
    replace_path=None,
    old_base_path=None
):
    if exclude_species is None:
        exclude_species = []

    data_folder_path = Path(data_folder_path)
    output_path = Path(output_path)
    split_file_path = Path(split_file_path)
    labelled_path = data_folder_path / "labelled"

    if not labelled_path.exists():
        raise ValueError(f"Labelled folder not found at {labelled_path}")
    if not split_file_path.exists():
        raise ValueError(f"Split file not found at {split_file_path}")

    # Validate species level
    if species_level not in ["l1", "l2", "l3", "l4"]:
        raise ValueError(
            f"Invalid species level: {species_level}. Must be one of: l1, l2, l3, l4"
        )

    species_col = f"species_{species_level}"

    train_datasets, val_datasets = load_split_from_file(split_file_path)

    print(f"\nScanning labelled folder for datasets...")
    all_datasets = get_all_dataset_names(labelled_path)
    print(f"Total unique datasets found: {len(all_datasets)}")

    train_set = set(train_datasets)
    val_set = set(val_datasets)
    test_datasets = list(all_datasets - train_set - val_set)

    print(f"\nDataset split summary:")
    print(f"  Train datasets: {len(train_datasets)}")
    print(f"  Val datasets: {len(val_datasets)}")
    print(f"  Test datasets: {len(test_datasets)}")

    overlap_train_val = train_set & val_set
    if overlap_train_val:
        print(f"WARNING: Overlapping datasets in train and val: {overlap_train_val}")

    missing_train = train_set - all_datasets
    missing_val = val_set - all_datasets
    if missing_train:
        print(f"WARNING: {len(missing_train)} train datasets not found in labelled folder")
    if missing_val:
        print(f"WARNING: {len(missing_val)} val datasets not found in labelled folder")

    all_metadata_files = list(labelled_path.rglob("*_metadata.csv"))
    if not all_metadata_files:
        raise ValueError(f"No metadata files found in {labelled_path}")

    print(f"\nFound {len(all_metadata_files)} metadata files")
    
    if replace_path:
        print(f"Using replacement path: {data_folder_path}")

    all_metadata = []
    for meta_file in all_metadata_files:
        df = pd.read_csv(meta_file)
        all_metadata.append(df)

    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    print(f"Total images in metadata: {len(combined_metadata)}")

    if flight_type != "both":
        combined_metadata = combined_metadata[combined_metadata["flight_type"] == flight_type]
        print(f"Images after filtering for {flight_type}: {len(combined_metadata)}")

    combined_metadata = combined_metadata[combined_metadata[species_col].notna()]
    print(f"Images with {species_level} labels: {len(combined_metadata)}")

    if exclude_species:
        print(f"\nExcluding species: {exclude_species}")
        images_before = len(combined_metadata)
        combined_metadata = combined_metadata[~combined_metadata[species_col].isin(exclude_species)]
        images_after = len(combined_metadata)
        print(f"Images removed: {images_before - images_after}")
        print(f"Images remaining: {images_after}")

    unique_classes = sorted(combined_metadata[species_col].unique())
    print(
        f"Unique classes at {species_level} level ({len(unique_classes)}): {unique_classes}"
    )

    train_data = combined_metadata[
        combined_metadata["dataset_name"].isin(train_datasets)
    ]
    val_data = combined_metadata[combined_metadata["dataset_name"].isin(val_datasets)]
    test_data = combined_metadata[combined_metadata["dataset_name"].isin(test_datasets)]

    print(f"\nImage counts per split:")
    print(f"  Train images: {len(train_data)}")
    print(f"  Val images: {len(val_data)}")
    print(f"  Test images: {len(test_data)}")

    print(f"\nFlight type distribution (filter: {flight_type}):")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        print(f"{split_name} split:")
        if len(split_data) > 0:
            print(split_data["flight_type"].value_counts())
        else:
            print("  No data")

    # Apply path fixes
    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()
    
    if replace_path is not None:
        train_data["image_path"] = train_data["image_path"].apply(lambda x: fix_image_path(x, data_folder_path, old_base_path))
        val_data["image_path"] = val_data["image_path"].apply(lambda x: fix_image_path(x, data_folder_path, old_base_path))
        test_data["image_path"] = test_data["image_path"].apply(lambda x: fix_image_path(x, data_folder_path, old_base_path))

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        if len(split_data) == 0:
            print(f"\nSkipping {split_name} split (no data)")
            continue

        split_path = output_path / split_name

        # Create class directories
        for class_name in unique_classes:
            class_dir = split_path / str(class_name)
            os.makedirs(class_dir, exist_ok=True)

        # Create hard links for images
        print(f"Creating hard links for {split_name} split...")
        missing_count = 0
        for _, row in tqdm(
            split_data.iterrows(),
            total=len(split_data),
            desc=f"Processing {split_name}",
        ):
            src_image_path = Path(row["image_path"])
            class_name = row[species_col]

            if not src_image_path.exists():
                missing_count += 1
                if missing_count <= 10:  # Only print first 10 warnings
                    print(f"WARNING: Source image not found: {src_image_path}")
                continue

            # Destination path in new structure
            dst_image_path = split_path / str(class_name) / src_image_path.name

            # Create hard link
            try:
                if not dst_image_path.exists():
                    os.link(src_image_path, dst_image_path)
            except Exception as e:
                print(f"ERROR creating hard link for {src_image_path}: {e}")
        
        if missing_count > 10:
            print(f"... and {missing_count - 10} more missing images")
        if missing_count > 0:
            print(f"Total missing images in {split_name}: {missing_count}")

    print(f"\nDataset structure created at: {output_path}")

    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        if len(split_data) == 0:
            continue
        print(f"\n{split_name} Split Class Distribution:")
        split_dist = split_data[species_col].value_counts().sort_index()
        for class_name, count in split_dist.items():
            print(f"  {class_name}: {count}")

    val_metadata_path = output_path / "val_metadata.csv"
    val_data.to_csv(val_metadata_path, index=False)
    print(f"\nCombined validation metadata saved to: {val_metadata_path}")

    test_metadata_path = output_path / "test_metadata.csv"
    test_data.to_csv(test_metadata_path, index=False)
    print(f"Combined test metadata saved to: {test_metadata_path}")


if __name__ == "__main__":
    create_mmpretrain_structure(
        data_folder_path=DATA_FOLDER_PATH,
        species_level=SPECIES_LEVEL,
        split_file_path=SPLIT_FILE_PATH,
        output_path=OUTPUT_PATH,
        flight_type=FLIGHT_TYPE,
        exclude_species=EXCLUDE_SPECIES,
        replace_path=REPLACE_PATH,
        old_base_path=OLD_BASE_PATH,
    )