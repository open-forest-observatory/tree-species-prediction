import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import _bootstrap
from configs.path_config import path_config

# Configuration parameters
DATA_FOLDER_PATH = Path("/ofo-share/project-data/species-prediction-project/intermediate/cropped_trees_live")
OUTPUT_PATH_L2 = Path("/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_nadir_2")
OUTPUT_PATH_L4 = Path("/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l4_nadir_2")
SPLIT_FILE_PATH = Path("/ofo-share/project-data/species-prediction-project/intermediate/train_val_split_l2.csv")
FLIGHT_TYPE = "nadir"  # Options: 'nadir', 'oblique', 'both'

# Classes to include at each level
L2_CLASSES = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']
L4_CLASSES = ['CUPR', 'FIR', 'HW', 'PI', 'TS']

# Species to explicitly exclude at L2 level (these trees will be excluded from both L2 and L4)
L2_EXCLUDE = ['QUKE', 'PIMO3', "ARME", "JUOC"]

# If the cropped tree images have moved from their original location at the point of creation,
# image_path in the metadata files will need to be updated to DATA_FOLDER_PATH. Else, set to None.
REPLACE_PATH = None
OLD_BASE_PATH = "/ofo-share/repos/david/tree-species-prediction/scratch/cropped_trees"


def load_split_from_file(split_file_path):
    """Load train/val split from CSV file."""
    split_df = pd.read_csv(split_file_path)
    train_datasets = split_df[split_df["split"] == "train"]["dataset"].tolist()
    val_datasets = split_df[split_df["split"] == "val"]["dataset"].tolist()

    print(f"Loaded from split file:")
    print(f"  Train datasets: {len(train_datasets)}")
    print(f"  Val datasets: {len(val_datasets)}")

    return train_datasets, val_datasets


def get_all_dataset_names(data_folder_path):
    """Get all dataset names from the data folder."""
    all_datasets = set()
    for item in data_folder_path.iterdir():
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
    relative_path = original_path.relative_to(old_base_path)
    new_path = Path(replace_path) / relative_path
    return str(new_path)


def create_hard_links_for_split(split_data, split_name, output_path, species_col, unique_classes):
    """
    Create hard links for images in a split.
    
    Args:
        split_data: DataFrame containing the split data
        split_name: Name of the split (train/val/test)
        output_path: Base output path for this level (L2 or L4)
        species_col: Column name for species classification
        unique_classes: List of unique class names
    """
    if len(split_data) == 0:
        print(f"  Skipping {split_name} split (no data)")
        return

    split_path = output_path / split_name

    # Create class directories
    for class_name in unique_classes:
        class_dir = split_path / str(class_name)
        os.makedirs(class_dir, exist_ok=True)

    # Create hard links for images
    print(f"  Creating hard links for {split_name} split...")
    missing_count = 0
    for _, row in tqdm(
        split_data.iterrows(),
        total=len(split_data),
        desc=f"  Processing {split_name}",
    ):
        src_image_path = Path(row["image_path"])
        class_name = row[species_col]

        if not src_image_path.exists():
            missing_count += 1
            if missing_count <= 10:  # Only print first 10 warnings
                print(f"    WARNING: Source image not found: {src_image_path}")
            continue

        # Destination path in new structure
        dst_image_path = split_path / str(class_name) / src_image_path.name

        # Create hard link
        try:
            if not dst_image_path.exists():
                os.link(src_image_path, dst_image_path)
        except Exception as e:
            print(f"    ERROR creating hard link for {src_image_path}: {e}")

    if missing_count > 10:
        print(f"    ... and {missing_count - 10} more missing images")
    if missing_count > 0:
        print(f"    Total missing images in {split_name}: {missing_count}")


def create_unified_datasets(
    data_folder_path,
    split_file_path,
    output_path_l2,
    output_path_l4,
    l2_classes,
    l4_classes,
    l2_exclude,
    flight_type="both",
    replace_path=None,
    old_base_path=None,
):
    """
    Create both L2 and L4 datasets simultaneously, ensuring identical tree sets.
    
    Args:
        data_folder_path: Path to the cropped trees data
        split_file_path: Path to the train/val split CSV
        output_path_l2: Output path for L2 dataset
        output_path_l4: Output path for L4 dataset
        l2_classes: List of classes to include at L2 level
        l4_classes: List of classes to include at L4 level
        l2_exclude: List of species codes to exclude at L2 level (removes from both L2 and L4)
        flight_type: Type of flight data to include
        replace_path: Whether to replace paths (True/False)
        old_base_path: Old base path to replace (if replace_path is True)
    """
    data_folder_path = Path(data_folder_path)
    output_path_l2 = Path(output_path_l2)
    output_path_l4 = Path(output_path_l4)
    split_file_path = Path(split_file_path)

    # Validate paths
    if not data_folder_path.exists():
        raise ValueError(f"Data folder not found at {data_folder_path}")
    if not split_file_path.exists():
        raise ValueError(f"Split file not found at {split_file_path}")

    print("="*80)
    print("UNIFIED L2 AND L4 DATASET CREATION (FULL DATASET)")
    print("="*80)

    # Load splits
    train_datasets, val_datasets = load_split_from_file(split_file_path)

    # Get all datasets
    print(f"\nScanning data folder for datasets...")
    all_datasets = get_all_dataset_names(data_folder_path)
    print(f"Total unique datasets found: {len(all_datasets)}")

    # Determine test datasets (those not in train or val)
    train_set = set(train_datasets)
    val_set = set(val_datasets)
    test_datasets = list(all_datasets - train_set - val_set)

    print(f"\nDataset split summary:")
    print(f"  Train datasets: {len(train_datasets)}")
    print(f"  Val datasets: {len(val_datasets)}")
    print(f"  Test datasets: {len(test_datasets)}")

    # Check for overlaps
    overlap_train_val = train_set & val_set
    if overlap_train_val:
        raise ValueError(f"Overlapping datasets in train and val: {overlap_train_val}")

    # Check for missing datasets
    missing_train = train_set - all_datasets
    missing_val = val_set - all_datasets
    if missing_train:
        print(f"WARNING: {len(missing_train)} train datasets not found in data folder")
    if missing_val:
        print(f"WARNING: {len(missing_val)} val datasets not found in data folder")

    # Load all metadata
    all_metadata_files = list(data_folder_path.rglob("*_metadata.csv"))
    if not all_metadata_files:
        raise ValueError(f"No metadata files found in {data_folder_path}")

    print(f"\nFound {len(all_metadata_files)} metadata files")
    print("Loading metadata...")

    if replace_path:
        print(f"Using replacement path: {data_folder_path}")

    all_metadata = []
    for meta_file in all_metadata_files:
        df = pd.read_csv(meta_file)
        all_metadata.append(df)

    # Store original metadata for exclusion tracking
    original_metadata = pd.concat(all_metadata, ignore_index=True)
    print(f"Total images in metadata: {len(original_metadata)}")

    # Start with all metadata
    combined_metadata = original_metadata.copy()

    # Filter by flight type
    if flight_type != "both":
        combined_metadata = combined_metadata[
            combined_metadata["flight_type"] == flight_type
        ]
        print(f"Images after filtering for {flight_type}: {len(combined_metadata)}")

    # Create global tree unique ID column
    print("\nCreating global_tree_unique_id column...")
    combined_metadata["global_tree_unique_id"] = (
        combined_metadata["dataset_name"].astype(str) + "_" + 
        combined_metadata["tree_unique_id"].astype(str)
    )
    
    # CRITICAL: Filter to trees that have BOTH valid L2 AND valid L4 labels
    print("\n" + "="*80)
    print("FILTERING TO TREES WITH VALID LABELS AT BOTH L2 AND L4")
    print("="*80)

    initial_count = len(combined_metadata)
    
    # Filter for non-null L2 labels
    combined_metadata = combined_metadata[combined_metadata["species_l2"].notna()]
    print(f"Images with L2 labels: {len(combined_metadata)}")
    
    # Filter for non-null L4 labels
    combined_metadata = combined_metadata[combined_metadata["species_l4"].notna()]
    print(f"Images with both L2 and L4 labels: {len(combined_metadata)}")

    # EXCLUDE trees with specific L2 species codes
    if l2_exclude:
        print(f"\nExcluding trees with L2 species: {l2_exclude}")
        trees_before_exclusion = len(combined_metadata["global_tree_unique_id"].unique())
        images_before_exclusion = len(combined_metadata)
        
        # Get tree IDs to exclude (any tree with these L2 codes)
        excluded_tree_ids = set(
            combined_metadata[combined_metadata["species_l2"].isin(l2_exclude)]["global_tree_unique_id"].unique()
        )
        
        # Remove ALL images of these trees (from both L2 and L4 datasets)
        combined_metadata = combined_metadata[
            ~combined_metadata["global_tree_unique_id"].isin(excluded_tree_ids)
        ]
        
        trees_after_exclusion = len(combined_metadata["global_tree_unique_id"].unique())
        images_after_exclusion = len(combined_metadata)
        
        print(f"Trees excluded: {trees_before_exclusion - trees_after_exclusion}")
        print(f"Images excluded: {images_before_exclusion - images_after_exclusion}")
        print(f"Remaining trees: {trees_after_exclusion}")
        print(f"Remaining images: {images_after_exclusion}")

    # Filter to only include specified classes at L2
    print(f"\nFiltering to L2 classes: {l2_classes}")
    l2_filtered = combined_metadata[combined_metadata["species_l2"].isin(l2_classes)]
    print(f"Images with valid L2 classes: {len(l2_filtered)}")
    
    # Filter to only include specified classes at L4
    print(f"\nFiltering to L4 classes: {l4_classes}")
    l4_filtered = combined_metadata[combined_metadata["species_l4"].isin(l4_classes)]
    print(f"Images with valid L4 classes: {len(l4_filtered)}")

    # Get the intersection: trees that have valid classes at BOTH levels
    print("\nFinding trees valid at BOTH L2 and L4...")
    valid_tree_ids_l2 = set(l2_filtered["global_tree_unique_id"].unique())
    valid_tree_ids_l4 = set(l4_filtered["global_tree_unique_id"].unique())
    valid_tree_ids = valid_tree_ids_l2 & valid_tree_ids_l4
    
    print(f"Trees with valid L2 classes: {len(valid_tree_ids_l2)}")
    print(f"Trees with valid L4 classes: {len(valid_tree_ids_l4)}")
    print(f"Trees valid at BOTH levels: {len(valid_tree_ids)}")
    
    # Trees only in one level
    only_l2 = valid_tree_ids_l2 - valid_tree_ids_l4
    only_l4 = valid_tree_ids_l4 - valid_tree_ids_l2
    
    if only_l2:
        print(f"Trees ONLY valid at L2 (will be excluded): {len(only_l2)}")
    if only_l4:
        print(f"Trees ONLY valid at L4 (will be excluded): {len(only_l4)}")

    # Filter combined metadata to only include these valid trees
    combined_metadata = combined_metadata[
        combined_metadata["global_tree_unique_id"].isin(valid_tree_ids)
    ]
    
    print(f"\nTotal images after filtering to common valid trees: {len(combined_metadata)}")
    print(f"Images removed: {initial_count - len(combined_metadata)}")

    # Save excluded trees information
    # Exclude based on original metadata, accounting for flight type filter
    if flight_type != "both":
        # Only consider exclusions from the flight-type-filtered data
        excluded_metadata = original_metadata[
            original_metadata["flight_type"] == flight_type
        ]
    else:
        excluded_metadata = original_metadata.copy()
    
    # Create global_tree_unique_id for excluded metadata
    excluded_metadata["global_tree_unique_id"] = (
        excluded_metadata["dataset_name"].astype(str) + "_" + 
        excluded_metadata["tree_unique_id"].astype(str)
    )
    
    # Now filter to excluded tree IDs
    excluded_metadata = excluded_metadata[
        ~excluded_metadata["global_tree_unique_id"].isin(valid_tree_ids)
    ]
    
    print(f"\nTotal images excluded: {len(excluded_metadata)}")
    print(f"Total unique trees excluded: {len(excluded_metadata['global_tree_unique_id'].unique())}")
    
    if len(excluded_metadata) > 0:
        print(f"\n" + "="*80)
        print("SAVING EXCLUDED TREES INFORMATION")
        print("="*80)
        
        for output_path, level_name in [(output_path_l2, "L2"), (output_path_l4, "L4")]:
            excluded_dir = output_path / "excluded_trees"
            os.makedirs(excluded_dir, exist_ok=True)
            
            print(f"\n{level_name} excluded trees:")
            for split_name, split_datasets in [
                ("train", train_datasets),
                ("val", val_datasets),
                ("test", test_datasets),
            ]:
                split_excluded = excluded_metadata[
                    excluded_metadata["dataset_name"].isin(split_datasets)
                ]
                
                if len(split_excluded) > 0:
                    excluded_info = split_excluded[
                        ["tree_unique_id", "species_l2", "species_l4", "dataset_name", "image_id"]
                    ].copy()
                    
                    excluded_csv_path = excluded_dir / f"{split_name}_excluded_trees.csv"
                    excluded_info.to_csv(excluded_csv_path, index=False)
                    
                    unique_trees = len(split_excluded["global_tree_unique_id"].unique())
                    print(f"  {split_name}: {len(split_excluded)} images ({unique_trees} unique trees) saved to {excluded_csv_path}")

    # Apply path fixes if needed
    if replace_path is not None:
        print(f"\nApplying path replacement...")
        combined_metadata["image_path"] = combined_metadata["image_path"].apply(
            lambda x: fix_image_path(x, data_folder_path, old_base_path)
        )

    # Split data by train/val/test
    train_data = combined_metadata[combined_metadata["dataset_name"].isin(train_datasets)].copy()
    val_data = combined_metadata[combined_metadata["dataset_name"].isin(val_datasets)].copy()
    test_data = combined_metadata[combined_metadata["dataset_name"].isin(test_datasets)].copy()

    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\nImage counts per split:")
    print(f"  Train images: {len(train_data)}")
    print(f"  Val images: {len(val_data)}")
    print(f"  Test images: {len(test_data)}")

    print(f"\nUnique tree counts per split:")
    print(f"  Train trees: {len(train_data['global_tree_unique_id'].unique())}")
    print(f"  Val trees: {len(val_data['global_tree_unique_id'].unique())}")
    print(f"  Test trees: {len(test_data['global_tree_unique_id'].unique())}")

    print(f"\nFlight type distribution (filter: {flight_type}):")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        if len(split_data) > 0:
            print(f"{split_name} split:")
            print(split_data["flight_type"].value_counts().to_string(header=False))

    # Create L2 dataset
    print("\n" + "="*80)
    print("CREATING L2 DATASET")
    print("="*80)
    
    l2_classes_in_data = sorted(train_data["species_l2"].unique())
    print(f"L2 classes in data ({len(l2_classes_in_data)}): {l2_classes_in_data}")
    
    print("\nL2 Class Distribution:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        if len(split_data) > 0:
            print(f"\n{split_name} split:")
            split_dist = split_data["species_l2"].value_counts().sort_index()
            for class_name, count in split_dist.items():
                print(f"  {class_name}: {count}")

    print(f"\nCreating L2 directory structure at: {output_path_l2}")
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        create_hard_links_for_split(
            split_data, split_name, output_path_l2, "species_l2", l2_classes_in_data
        )

    # Save L2 metadata
    val_metadata_path_l2 = output_path_l2 / "val_metadata.csv"
    val_data.to_csv(val_metadata_path_l2, index=False)
    print(f"\nL2 validation metadata saved to: {val_metadata_path_l2}")

    test_metadata_path_l2 = output_path_l2 / "test_metadata.csv"
    test_data.to_csv(test_metadata_path_l2, index=False)
    print(f"L2 test metadata saved to: {test_metadata_path_l2}")

    # Create L4 dataset
    print("\n" + "="*80)
    print("CREATING L4 DATASET")
    print("="*80)
    
    l4_classes_in_data = sorted(train_data["species_l4"].unique())
    print(f"L4 classes in data ({len(l4_classes_in_data)}): {l4_classes_in_data}")
    
    print("\nL4 Class Distribution:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        if len(split_data) > 0:
            print(f"\n{split_name} split:")
            split_dist = split_data["species_l4"].value_counts().sort_index()
            for class_name, count in split_dist.items():
                print(f"  {class_name}: {count}")

    print(f"\nCreating L4 directory structure at: {output_path_l4}")
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        create_hard_links_for_split(
            split_data, split_name, output_path_l4, "species_l4", l4_classes_in_data
        )

    # Save L4 metadata
    val_metadata_path_l4 = output_path_l4 / "val_metadata.csv"
    val_data.to_csv(val_metadata_path_l4, index=False)
    print(f"\nL4 validation metadata saved to: {val_metadata_path_l4}")

    test_metadata_path_l4 = output_path_l4 / "test_metadata.csv"
    test_data.to_csv(test_metadata_path_l4, index=False)
    print(f"L4 test metadata saved to: {test_metadata_path_l4}")

    # Final verification
    print("\n" + "="*80)
    print("VERIFICATION: TREE COUNT CONSISTENCY")
    print("="*80)
    
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        if len(split_data) > 0:
            unique_trees = len(split_data["global_tree_unique_id"].unique())
            print(f"{split_name}: {unique_trees} unique trees (same in both L2 and L4)")

    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    create_unified_datasets(
        data_folder_path=DATA_FOLDER_PATH,
        split_file_path=SPLIT_FILE_PATH,
        output_path_l2=OUTPUT_PATH_L2,
        output_path_l4=OUTPUT_PATH_L4,
        l2_classes=L2_CLASSES,
        l4_classes=L4_CLASSES,
        l2_exclude=L2_EXCLUDE,
        flight_type=FLIGHT_TYPE,
        replace_path=REPLACE_PATH,
        old_base_path=OLD_BASE_PATH,
    )