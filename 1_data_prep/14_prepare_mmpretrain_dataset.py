import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

import _bootstrap
from configs.path_config import path_config

# Configuration parameters
DATA_FOLDER_PATH = path_config.cropped_tree_training_images
OUTPUT_PATH = path_config.mmpretrain_dataset_folder
SPECIES_LEVEL = "l2"  # Class lumping level options: 'l1', 'l2', 'l3', 'l4'

# TODO: Handle merging and saving all validation metadata files as one. This is needed by mmpretrain to calculate tree_acc

# TODO: Update these lists based on path_config.train_test_split_file
TRAIN_DATASETS = [
    "0074_000874_000932",
    "0076_000874_000932",
    "0078_000810_000808",
    "0079_000810_000808",
    "0080_000810_000808"
]

VAL_DATASETS = [
    "0073_000874_000932",
    "0077_000810_000808"
]


def create_mmpretrain_structure(
    data_folder_path,
    species_level,
    train_datasets,
    val_datasets,
    output_path
):
    """
    Create this MMPretrain-compatible folder structure:
    output_path/
        train/
            class_1/
                img1.jpg
                img2.jpg
            class_2/
                img1.jpg
                img2.jpg
        val/
            class_1/
                img1.jpg
                img2.jpg
            class_2/
                img1.jpg
                img2.jpg
    
    Args:
        data_folder_path (Path): Path to the data folder containing 'labelled' subdirectory
        species_level (str): Species lumping level ('l1', 'l2', 'l3', or 'l4')
        train_datasets (list): List of dataset names for training split
        val_datasets (list): List of dataset names for validation split
        output_path (Path): Path where the new structure will be created
    """
    
    data_folder_path = Path(data_folder_path)
    output_path = Path(output_path)
    labelled_path = data_folder_path / "labelled"
    
    if not labelled_path.exists():
        raise ValueError(f"Labelled folder not found at {labelled_path}")
    
    # Validate species level
    if species_level not in ['l1', 'l2', 'l3', 'l4']:
        raise ValueError(f"Invalid species level: {species_level}. Must be one of: l1, l2, l3, l4")
    
    species_col = f"species_{species_level}"
    
    # Collect all metadata files
    all_metadata_files = list(labelled_path.rglob("*_metadata.csv"))
    
    if not all_metadata_files:
        raise ValueError(f"No metadata files found in {labelled_path}")
    
    print(f"Found {len(all_metadata_files)} metadata files")
    
    # Load and combine all metadata
    all_metadata = []
    for meta_file in all_metadata_files:
        df = pd.read_csv(meta_file)
        all_metadata.append(df)
    
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    print(f"Total images in metadata: {len(combined_metadata)}")
    
    # Filter to only include rows with non-null species labels for the specified level
    combined_metadata = combined_metadata[combined_metadata[species_col].notna()]
    print(f"Images with {species_level} labels: {len(combined_metadata)}")
    
    # Get unique classes
    unique_classes = sorted(combined_metadata[species_col].unique())
    print(f"Unique classes at {species_level} level ({len(unique_classes)}): {unique_classes}")
    
    # Split data into train and val
    train_data = combined_metadata[combined_metadata['dataset_name'].isin(train_datasets)]
    val_data = combined_metadata[combined_metadata['dataset_name'].isin(val_datasets)]
    
    print(f"Train images: {len(train_data)}")
    print(f"Val images: {len(val_data)}")
    
    # Create directory structure
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        split_path = output_path / split_name
        
        # Create class directories
        for class_name in unique_classes:
            class_dir = split_path / str(class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # Create hard links for images
        print(f"Creating hard links for {split_name} split...")
        for _, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Processing {split_name}"):
            src_image_path = Path(row['image_path'])
            class_name = row[species_col]
            
            if not src_image_path.exists():
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
    
    print(f"Dataset structure created at: {output_path}")
    
    # Print class distribution
    print("Train Split Class Distribution:")
    train_dist = train_data[species_col].value_counts().sort_index()
    for class_name, count in train_dist.items():
        print(f"{class_name}: {count}")
    
    print("Val Split Class Distribution:")
    val_dist = val_data[species_col].value_counts().sort_index()
    for class_name, count in val_dist.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    create_mmpretrain_structure(
        data_folder_path=DATA_FOLDER_PATH,
        species_level=SPECIES_LEVEL,
        train_datasets=TRAIN_DATASETS,
        val_datasets=VAL_DATASETS,
        output_path=OUTPUT_PATH
    )