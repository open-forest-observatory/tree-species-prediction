import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Note: This script should be run with the `open-mmlab` environment activated
from mmpretrain.apis import ImageClassificationInferencer
from tqdm import tqdm

import _bootstrap
from configs.path_config import path_config

INPUT_ROOT = path_config.cropped_tree_training_images / "labelled"
OUTPUT_ROOT = path_config.live_cropped_trees
CONFIG_FILE = path_config.dead_live_model_config
CHECKPOINT_FILE = path_config.dead_live_model_checkpoint
DEVICE = "cuda"
FIX_IMAGE_PATHS = True  # If True, fix image paths to point to current dataset structure

# Class index mapping
IDX_TO_CLASS = {
    0: "Dead",
    1: "Live",
}

# Initialize the MMPretrain inferencer
inferencer = ImageClassificationInferencer(
    model=str(CONFIG_FILE),
    pretrained=str(CHECKPOINT_FILE),
    device=DEVICE
)

# Tree-level aggregation to get Dead/Live labels at the tree level based on majority vote
def aggregate_tree_predictions(pred_labels):
    """
    Args:
        pred_labels (List[int]): List of predicted labels (0 or 1) for images of a single tree.
    Returns:
        str: "Live" or "Dead" based on majority vote.
    """
    counts = Counter(pred_labels)

    # Majority vote
    if counts[1] > counts[0]:
        return "Live"
    else:
        return "Dead"


if __name__ == "__main__":

    # Ensure output root directory exists
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_dir in sorted(INPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue

        # Get the dataset name
        dataset_name = dataset_dir.name
        print(f"Processing dataset: {dataset_name}")

        metadata_csv = dataset_dir / f"{dataset_name}_metadata.csv"
        if not metadata_csv.exists():
            print(f"Missing metadata CSV, skipping")
            continue

        # Load the metadata for the dataset
        metadata_df = pd.read_csv(metadata_csv)

        # Group by tree
        trees = metadata_df.groupby("tree_unique_id")

        live_rows = []
        dead_tree_ids = []
        output_dataset_dir = OUTPUT_ROOT / dataset_name

        # Iteration of trees yields (tree_id, tree_df) pairs where 
        # tree_id is the group key and tree_df is a DataFrame of rows for that tree.
        for tree_id, tree_df in tqdm(trees, desc=" Trees"):

            # Pad tree ID with zeros
            tree_id_str = str(tree_id).zfill(5)

            if FIX_IMAGE_PATHS:
                image_paths = [
                    dataset_dir / f"treeID{tree_id_str}" / Path(p).name
                    for p in tree_df["image_path"]
                ]
            else:
                image_paths = [Path(p) for p in tree_df["image_path"]]
            
            # Run inference on batched images of this tree
            results = inferencer(image_paths)

            pred_labels = [
                int(r["pred_label"]) for r in results
            ]

            # Aggregate to get tree-level label
            tree_label = aggregate_tree_predictions(pred_labels)

            # Skip dead trees
            if tree_label != "Live":
                dead_tree_ids.append(tree_id_str)
                continue

            # Create output tree folder
            tree_dir_name = f"treeID{tree_id_str}"
            src_tree_dir = dataset_dir / tree_dir_name
            dst_tree_dir = output_dataset_dir / tree_dir_name
            dst_tree_dir.mkdir(parents=True, exist_ok=True)

            # Hardlink images
            for img_path in image_paths:
                img_path = Path(img_path)
                dst_img = dst_tree_dir / img_path.name

                if not dst_img.exists():
                    os.link(img_path, dst_img)

            # Keep metadata rows
            tree_df = tree_df.copy()
            # Update image paths to new location in live cropped trees folder
            tree_df["image_path"] = [
                str(dst_tree_dir / Path(p).name)
                for p in tree_df["image_path"]
            ]

            live_rows.append(tree_df)

        if not live_rows:
            print("No live trees found")
            continue

        # Write filtered metadata
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        # At the dataset level, concatenate all live tree rows into a single dataframe
        live_df = pd.concat(live_rows, ignore_index=True)
        # Create output CSV to save live trees metadata
        out_csv = output_dataset_dir / f"{dataset_name}_metadata.csv"
        live_df.to_csv(out_csv, index=False)

        # Write dead tree IDs to a separate file
        dead_tree_file = output_dataset_dir / f"{dataset_name}_dead_tree_ids.txt"
        with open(dead_tree_file, "w") as f:
            for tid in dead_tree_ids:
                f.write(f"{tid}\n")

        print(f"Saved {len(live_df)} image rows to {out_csv}")
        print(f"Saved {len(dead_tree_ids)} dead tree IDs to {dead_tree_file}")

    print("All datasets processed.")
