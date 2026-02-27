"""
Remap per-chip live/dead predictions to per-tree outputs.

Reads per-chip predictions (Live/Dead) from the chipping-and-live-dead Argo workflow,
maps chip render IDs to real tree IDs via IDs_to_labels.json, aggregates predictions
per tree by majority vote, and writes live-tree chips to a structured output directory.

Run with the TRAM-dev conda environment:
    /home/exouser/miniconda3/envs/TRAM-dev/bin/python remap_live_dead.py
"""

import json
import shutil
import uuid
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PREDICTIONS_DIR = Path(
    "/ofo-share/argo-data/argo-output/chipping-and-live-dead/per_chip_predictions"
)
RENDERS_DIR = Path(
    "/ofo-share/argo-data/argo-input/chipping-and-live-dead/renders"
)
GROUND_DATA_DIR = Path(
    "/ofo-share/project-data/species-prediction-project/intermediate"
    "/drone_crowns_with_field_attributes"
)
SPECIES_CROSSWALK_PATH = Path(
    "/ofo-share/project-data/species-prediction-project/intermediate"
    "/species_class_crosswalk.csv"
)
OUTPUT_DIR = Path(
    "/ofo-share/project-data/species-prediction-project/intermediate"
    "/cropped_trees_live_david"
)

# Chip paths in the JSON use /data/ as a prefix; actual files live under /ofo-share/argo-data/
CHIP_PATH_PREFIX_IN_JSON = "/data/"
CHIP_PATH_PREFIX_REAL = "/ofo-share/argo-data/"

# Base of the chips directory (used to reconstruct the source image path)
CHIPS_BASE_REAL = "/ofo-share/argo-data/argo-output/chipping-and-live-dead/chips/"
SOURCE_IMAGES_BASE = "/ofo-share/argo-data/argo-input/datasets/"


# ---------------------------------------------------------------------------
# Species mapping helpers (mirroring 11_crop_trees.py)
# ---------------------------------------------------------------------------


def load_species_mappings(crosswalk_path: Path) -> dict:
    """Load species crosswalk and return per-level mapping dicts."""
    crosswalk_df = pd.read_csv(crosswalk_path)
    all_mappings = {}
    for level in ["l1", "l2", "l3", "l4"]:
        primary_col = f"primary_species_{level}"
        species_col = f"species_code_{level}"
        level_mapping = {}
        for _, row in crosswalk_df.iterrows():
            if row[primary_col]:
                level_mapping[row["species_code"]] = row[species_col]
        all_mappings[level] = level_mapping
    return all_mappings


def map_species_all_levels(original_species: str | None, all_mappings: dict) -> dict:
    """Map an original species code to all taxonomy levels."""
    if original_species is None:
        return {level: None for level in ["l1", "l2", "l3", "l4"]}
    return {
        level: all_mappings[level].get(original_species)
        for level in ["l1", "l2", "l3", "l4"]
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def real_chip_path(json_chip_path: str) -> Path:
    """Convert the /data/-prefixed path in the JSON to the real filesystem path."""
    return Path(json_chip_path.replace(CHIP_PATH_PREFIX_IN_JSON, CHIP_PATH_PREFIX_REAL, 1))


def source_image_path(json_chip_path: str) -> str:
    """
    Reconstruct the source raw image path from a chip path.

    e.g. /data/argo-output/.../chips/<plot>/<flight>/.../<src_id>/<num>.png
      -> /ofo-share/argo-data/argo-input/datasets/<plot>/<flight>/.../<src_id>.JPG
    """
    real = real_chip_path(json_chip_path)
    # relative path from the chips base: <plot>/<flight_type>/.../src_id/<num>.png
    rel = real.relative_to(CHIPS_BASE_REAL)
    # drop the chip number filename, keep the source image dir as a file with .JPG
    src_rel = rel.parent.with_suffix(".JPG")
    return SOURCE_IMAGES_BASE + str(src_rel)


def flight_type_from_path(json_chip_path: str) -> str:
    """Extract 'nadir' or 'oblique' from the chip path."""
    real = real_chip_path(json_chip_path)
    rel = real.relative_to(CHIPS_BASE_REAL)
    # rel parts: (<plot_id>, <flight_type>, ...)
    return rel.parts[1]


def random_image_id() -> str:
    """Generate a random 10-digit image ID (matching 11_crop_trees.py convention)."""
    return str(uuid.uuid4().int)[:10]


# ---------------------------------------------------------------------------
# Per-plot processing
# ---------------------------------------------------------------------------


def process_plot(plot_name: str, all_species_mappings: dict) -> dict:
    """
    Process one plot end-to-end.

    Returns a stats dict with keys: trees_live, trees_dead, chips_copied.
    """
    stats = {"trees_live": 0, "trees_dead": 0, "chips_copied": 0}

    # -- Load predictions --
    predictions_path = PREDICTIONS_DIR / f"{plot_name}.json"
    with open(predictions_path) as f:
        chip_predictions = json.load(f)  # {chip_path: "Live"|"Dead"}

    # -- Load IDs_to_labels (chip render number -> 5-digit tree ID string) --
    ids_to_labels_path = RENDERS_DIR / plot_name / "IDs_to_labels.json"
    with open(ids_to_labels_path) as f:
        ids_to_labels = json.load(f)  # {"1": "00295", ...}

    # -- Load ground data for species info --
    ground_data_path = GROUND_DATA_DIR / f"{plot_name}.gpkg"
    plot_gdf = gpd.read_file(ground_data_path)[["unique_ID", "species_code"]]

    # -- Aggregate chip predictions per tree --
    # tree_votes: {tree_id_str: {"Live": int, "Dead": int}}
    tree_votes: dict[str, dict[str, int]] = {}
    # tree_chips: {tree_id_str: [json_chip_path, ...]}
    tree_chips: dict[str, list[str]] = {}

    for chip_path, prediction in chip_predictions.items():
        chip_num = Path(chip_path).stem  # e.g. "31"
        if chip_num not in ids_to_labels:
            continue
        tree_id = ids_to_labels[chip_num]  # e.g. "00499"

        if tree_id not in tree_votes:
            tree_votes[tree_id] = {"Live": 0, "Dead": 0}
            tree_chips[tree_id] = []

        tree_votes[tree_id][prediction] += 1
        tree_chips[tree_id].append(chip_path)

    # -- Determine live vs dead trees by majority vote --
    dead_tree_ids = []
    live_tree_ids = []
    for tree_id, votes in tree_votes.items():
        if votes["Dead"] > votes["Live"]:
            dead_tree_ids.append(tree_id)
        else:
            live_tree_ids.append(tree_id)

    stats["trees_live"] = len(live_tree_ids)
    stats["trees_dead"] = len(dead_tree_ids)

    # -- Set up output directory --
    plot_out_dir = OUTPUT_DIR / plot_name
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    # -- Copy live-tree chips and build metadata records --
    metadata_records = []

    for tree_id in live_tree_ids:
        tree_out_dir = plot_out_dir / f"treeID{tree_id}"
        tree_out_dir.mkdir(exist_ok=True)

        # Get species info
        species_row = plot_gdf.loc[plot_gdf["unique_ID"] == tree_id, "species_code"]
        original_species = species_row.iloc[0] if len(species_row) > 0 else None
        if pd.isna(original_species):
            original_species = None
        mapped = map_species_all_levels(original_species, all_species_mappings)

        for chip_path in tree_chips[tree_id]:
            src = real_chip_path(chip_path)
            if not src.exists():
                continue

            img_id = random_image_id()
            dst = tree_out_dir / f"{img_id}.png"
            shutil.copy2(src, dst)
            stats["chips_copied"] += 1

            metadata_records.append(
                {
                    "image_id": img_id,
                    "image_path": str(dst),
                    "tree_unique_id": tree_id,
                    "species_original": original_species,
                    "species_l1": mapped["l1"],
                    "species_l2": mapped["l2"],
                    "species_l3": mapped["l3"],
                    "species_l4": mapped["l4"],
                    "dataset_name": plot_name,
                    "flight_type": flight_type_from_path(chip_path),
                    "source_image_path": source_image_path(chip_path),
                }
            )

    # -- Write dead tree IDs --
    dead_ids_path = plot_out_dir / f"{plot_name}_dead_tree_ids.txt"
    with open(dead_ids_path, "w") as f:
        for tree_id in sorted(dead_tree_ids):
            f.write(f"{tree_id}\n")

    # -- Write metadata CSV --
    metadata_path = plot_out_dir / f"{plot_name}_metadata.csv"
    columns = [
        "image_id",
        "image_path",
        "tree_unique_id",
        "species_original",
        "species_l1",
        "species_l2",
        "species_l3",
        "species_l4",
        "dataset_name",
        "flight_type",
        "source_image_path",
    ]
    pd.DataFrame(metadata_records, columns=columns).to_csv(metadata_path, index=False)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_species_mappings = load_species_mappings(SPECIES_CROSSWALK_PATH)

    plot_names = sorted(p.stem for p in PREDICTIONS_DIR.glob("*.json"))
    print(f"Processing {len(plot_names)} plots...")

    total_live = total_dead = total_chips = 0

    for i, plot_name in enumerate(plot_names, 1):
        print(f"[{i}/{len(plot_names)}] {plot_name}", flush=True)
        try:
            stats = process_plot(plot_name, all_species_mappings)
            total_live += stats["trees_live"]
            total_dead += stats["trees_dead"]
            total_chips += stats["chips_copied"]
            print(f"  live={stats['trees_live']} dead={stats['trees_dead']} chips={stats['chips_copied']}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    print(f"\nDone.")
    print(f"  Live trees : {total_live}")
    print(f"  Dead trees : {total_dead}")
    print(f"  Chips copied: {total_chips}")